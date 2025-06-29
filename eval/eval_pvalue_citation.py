# mog/eval/eval_pvalue_citation.py
import os
import json
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import argparse
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_metrics_from_json(file_path: str) -> Dict:
    """Load metrics from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return {}


def extract_citation_scores(result_json: Dict) -> Dict[str, float]:
    """Extract citation evaluation scores from result JSON."""
    metrics = {}
    for article_id, article_data in result_json.items():
        for metric in ["citation_rec", "citation_prec", "sent_uncitated_rate"]:
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(article_data[metric])

    return metrics


def compute_pvalue(
    mog_scores: List[float], baseline_scores: List[float], alternative: str = "greater"
) -> float:
    """
    Compute p-value for paired t-test between MOG and baseline scores.

    Args:
        mog_scores: List of scores for MOG model
        baseline_scores: List of scores for baseline model
        alternative: The alternative hypothesis, either 'two-sided', 'less', or 'greater'
                    'greater' means MOG > baseline is the alternative hypothesis

    Returns:
        p-value of the statistical test
    """
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(
        mog_scores, baseline_scores, alternative=alternative
    )
    return p_value


def numpy_to_python_type(obj: Any) -> Any:
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def analyze_citation_metrics(
    mog_results: Dict, baseline_results: Dict, save_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze p-values for citation metrics between MOG and baseline.

    Args:
        mog_results: Citation evaluation results for MOG model
        baseline_results: Citation evaluation results for baseline model
        save_path: Optional path to save results

    Returns:
        Dictionary with p-values and mean differences for each metric
    """
    # Ensure we're comparing the same articles
    common_articles = set(mog_results.keys()) & set(baseline_results.keys())
    if len(common_articles) < len(mog_results) or len(common_articles) < len(
        baseline_results
    ):
        logging.warning(
            f"Found only {len(common_articles)} common articles out of "
            f"{len(mog_results)} MOG articles and {len(baseline_results)} baseline articles"
        )

    # Filter to common articles
    mog_filtered = {k: v for k, v in mog_results.items() if k in common_articles}
    baseline_filtered = {
        k: v for k, v in baseline_results.items() if k in common_articles
    }

    results = {}
    # Citation metrics we care about
    original_metrics = ["citation_rec", "citation_prec", "sent_uncitated_rate"]

    for metric in original_metrics:
        # 对于sent_uncitated_rate，我们计算cited_rate = 100 - sent_uncitated_rate
        is_uncitated_metric = metric == "sent_uncitated_rate"
        result_metric = "sent_cited_rate" if is_uncitated_metric else metric

        # 获取原始分数
        mog_raw_scores = [
            article_data[metric] for article_data in mog_filtered.values()
        ]
        baseline_raw_scores = [
            article_data[metric] for article_data in baseline_filtered.values()
        ]

        # 如果是uncitated_rate，转换为cited_rate
        if is_uncitated_metric:
            mog_scores = [100 - score for score in mog_raw_scores]
            baseline_scores = [100 - score for score in baseline_raw_scores]
            # 对于cited_rate，更高更好，使用"greater"
            alternative = "greater"
        else:
            mog_scores = mog_raw_scores
            baseline_scores = baseline_raw_scores
            alternative = "greater"

        # Paired t-test
        p_value = compute_pvalue(mog_scores, baseline_scores, alternative)

        # Mean difference
        mean_diff = np.mean(mog_scores) - np.mean(baseline_scores)

        results[result_metric] = {
            "p_value": numpy_to_python_type(p_value),
            "mean_difference": numpy_to_python_type(mean_diff),
            "mog_mean": numpy_to_python_type(np.mean(mog_scores)),
            "baseline_mean": numpy_to_python_type(np.mean(baseline_scores)),
            "significant": numpy_to_python_type(p_value < 0.05),
            "sample_size": len(common_articles),
        }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def find_best_baseline_per_metric(
    all_baselines_results: Dict[str, Dict[str, Dict]],
) -> Dict[str, str]:
    """
    Find the best baseline for each citation metric.

    Args:
        all_baselines_results: Dictionary mapping baseline names to their results per metric

    Returns:
        Dictionary mapping metrics to their best baseline name
    """
    best_baselines = {}

    # First, identify all available metrics
    all_metrics = set()
    for baseline_results in all_baselines_results.values():
        all_metrics.update(baseline_results.keys())

    # For each metric, find the best baseline
    for metric in all_metrics:
        # 所有指标都是越高越好
        best_score = -float("inf")
        best_baseline = None

        for baseline_name, results in all_baselines_results.items():
            if metric in results:
                score = results[metric]["baseline_mean"]
                if score > best_score:
                    best_score = score
                    best_baseline = baseline_name

        if best_baseline:
            best_baselines[metric] = best_baseline

    return best_baselines


def compare_with_best_baselines_per_metric(
    mog_path: str, baseline_paths: Dict[str, str], output_dir: str
) -> None:
    """
    Compare MOG against the best baseline for each citation metric.

    Args:
        mog_path: Path to MOG citation evaluation results JSON
        baseline_paths: Dictionary mapping baseline names to their result JSON paths
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load MOG results
    mog_results = load_metrics_from_json(mog_path)
    if not mog_results:
        logging.error(f"Failed to load MOG results from {mog_path}")
        return

    # Load all baseline results first
    baseline_results_dict = {}
    all_metrics_results = {}

    for baseline_name, baseline_path in baseline_paths.items():
        baseline_results = load_metrics_from_json(baseline_path)
        if not baseline_results:
            logging.error(
                f"Failed to load {baseline_name} results from {baseline_path}"
            )
            continue

        baseline_results_dict[baseline_name] = baseline_results

        # Run analysis for each baseline to get metrics
        metrics_results = analyze_citation_metrics(mog_results, baseline_results)
        all_metrics_results[baseline_name] = metrics_results

    # Find the best baseline for each metric
    best_baselines_per_metric = find_best_baseline_per_metric(all_metrics_results)

    if not best_baselines_per_metric:
        logging.error(
            "Could not determine best baselines. No valid metrics or baselines found."
        )
        return

    # Prepare results for saving
    final_results = {}

    print("\n=== MOG vs Best Baseline per Citation Metric ===")

    # Compare MOG with the best baseline for each metric
    for metric, best_baseline in best_baselines_per_metric.items():
        # Get the results already computed for this baseline
        results_for_metric = all_metrics_results[best_baseline][metric]

        # Add the baseline name to the results
        results_for_metric["best_baseline"] = best_baseline
        final_results[metric] = results_for_metric

        # Print summary for this metric
        significance = (
            "SIGNIFICANT" if results_for_metric["significant"] else "not significant"
        )
        print(
            f"{metric}: best baseline = {best_baseline}, p={results_for_metric['p_value']:.4f} "
            f"(MOG: {results_for_metric['mog_mean']:.4f}, {best_baseline}: {results_for_metric['baseline_mean']:.4f}) - {significance}"
        )

    # Save the final results
    save_path = os.path.join(output_dir, "citation_pvalue.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    logging.info(
        f"P-value analysis completed for MOG vs best baselines per citation metric."
    )
    logging.info(f"Results saved to {save_path}")


def generate_comparison_table(
    mog_path: str, baseline_paths: Dict[str, str], output_dir: str
) -> None:
    """
    Generate a comparison table of all citation metrics across all models.

    Args:
        mog_path: Path to MOG citation evaluation results JSON
        baseline_paths: Dictionary mapping baseline names to their result JSON paths
        output_dir: Directory to save results
    """
    # Load MOG results
    mog_results = load_metrics_from_json(mog_path)
    if not mog_results:
        logging.error(f"Failed to load MOG results from {mog_path}")
        return

    # Prepare data for the table
    results_data = []

    # First, get the average values for MOG
    mog_metrics = {"citation_rec": [], "citation_prec": [], "sent_uncitated_rate": []}

    for article_id, article_data in mog_results.items():
        for metric in mog_metrics.keys():
            mog_metrics[metric].append(article_data[metric])

    # 转换sent_uncitated_rate为sent_cited_rate
    mog_means = {
        "citation_rec": np.mean(mog_metrics["citation_rec"]),
        "citation_prec": np.mean(mog_metrics["citation_prec"]),
        "sent_cited_rate": 100 - np.mean(mog_metrics["sent_uncitated_rate"]),
    }

    # Now process each baseline
    for baseline_name, baseline_path in baseline_paths.items():
        baseline_results = load_metrics_from_json(baseline_path)
        if not baseline_results:
            logging.error(
                f"Failed to load {baseline_name} results from {baseline_path}"
            )
            continue

        # Calculate means for this baseline
        baseline_metrics = {
            "citation_rec": [],
            "citation_prec": [],
            "sent_uncitated_rate": [],
        }

        for article_id, article_data in baseline_results.items():
            for metric in baseline_metrics.keys():
                baseline_metrics[metric].append(article_data[metric])

        # 转换sent_uncitated_rate为sent_cited_rate
        baseline_means = {
            "citation_rec": np.mean(baseline_metrics["citation_rec"]),
            "citation_prec": np.mean(baseline_metrics["citation_prec"]),
            "sent_cited_rate": 100 - np.mean(baseline_metrics["sent_uncitated_rate"]),
        }

        # Calculate p-values for MOG vs this baseline
        metrics_results = analyze_citation_metrics(mog_results, baseline_results)

        # Add data for each metric in metrics_results
        for metric in metrics_results.keys():
            mog_mean = mog_means.get(metric, 0)
            baseline_mean = baseline_means.get(metric, 0)

            row = {
                "Metric": metric,
                "MOG": mog_mean,
                f"{baseline_name}": baseline_mean,
                f"p-value (MOG vs {baseline_name})": metrics_results[metric]["p_value"],
                f"Significant (MOG vs {baseline_name})": metrics_results[metric][
                    "significant"
                ],
            }
            results_data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "citation_pvalue_comparison.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Citation comparison table saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute p-values for citation metrics between MOG and baseline models"
    )
    parser.add_argument(
        "--mog_results", required=True, help="Path to MOG citation evaluation JSON file"
    )
    parser.add_argument(
        "--baseline_results",
        required=True,
        nargs="+",
        help="Paths to baseline citation evaluation JSON files in format 'name:path'",
    )
    parser.add_argument(
        "--output_dir", default="pvalue_results", help="Directory to save result files"
    )
    parser.add_argument(
        "--generate_table",
        action="store_true",
        help="Generate a comparison table across all models",
    )

    args = parser.parse_args()

    # Parse baseline paths
    baseline_paths = {}
    for baseline_arg in args.baseline_results:
        name, path = baseline_arg.split(":", 1)
        baseline_paths[name] = path

    # Generate comparison table if requested
    if args.generate_table:
        generate_comparison_table(args.mog_results, baseline_paths, args.output_dir)

    # Compare with best baselines per metric
    compare_with_best_baselines_per_metric(
        args.mog_results, baseline_paths, args.output_dir
    )
