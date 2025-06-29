# mog/eval/eval_pvalue.py
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


def extract_topic_scores(detailed_json: Dict, metric_name: str) -> List[float]:
    """Extract scores for a specific metric across all topics."""
    return [topic_data[metric_name] for topic_data in detailed_json.values()]


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


def analyze_all_metrics(
    mog_detailed: Dict, baseline_detailed: Dict, save_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze p-values for all metrics between MOG and baseline.

    Args:
        mog_detailed: Detailed metrics for MOG model
        baseline_detailed: Detailed metrics for baseline model
        save_path: Optional path to save results

    Returns:
        Dictionary with p-values and mean differences for each metric
    """
    # Ensure we're comparing the same topics
    common_topics = set(mog_detailed.keys()) & set(baseline_detailed.keys())
    if len(common_topics) < len(mog_detailed) or len(common_topics) < len(
        baseline_detailed
    ):
        logging.warning(
            f"Found only {len(common_topics)} common topics out of "
            f"{len(mog_detailed)} MOG topics and {len(baseline_detailed)} baseline topics"
        )

    # Filter to common topics
    mog_filtered = {k: v for k, v in mog_detailed.items() if k in common_topics}
    baseline_filtered = {
        k: v for k, v in baseline_detailed.items() if k in common_topics
    }

    results = {}
    # Get all metrics from the first topic entry
    sample_topic = next(iter(mog_filtered.values()))
    metrics = sample_topic.keys()

    for metric in metrics:
        mog_scores = [topic_data[metric] for topic_data in mog_filtered.values()]
        baseline_scores = [
            topic_data[metric] for topic_data in baseline_filtered.values()
        ]

        # Paired t-test for MOG > baseline
        p_value = compute_pvalue(mog_scores, baseline_scores, "greater")

        # Mean difference
        mean_diff = np.mean(mog_scores) - np.mean(baseline_scores)

        results[metric] = {
            "p_value": numpy_to_python_type(p_value),
            "mean_difference": numpy_to_python_type(mean_diff),
            "mog_mean": numpy_to_python_type(np.mean(mog_scores)),
            "baseline_mean": numpy_to_python_type(np.mean(baseline_scores)),
            "significant": numpy_to_python_type(p_value < 0.05),
            "sample_size": len(common_topics),
        }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def find_best_baseline_per_metric(
    all_baselines_results: Dict[str, Dict[str, Dict]],
) -> Dict[str, str]:
    """
    Find the best baseline for each metric.

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
    Compare MOG against the best baseline for each metric.

    Args:
        mog_path: Path to MOG detailed results JSON
        baseline_paths: Dictionary mapping baseline names to their result JSON paths
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load MOG results
    mog_detailed = load_metrics_from_json(mog_path)
    if not mog_detailed:
        logging.error(f"Failed to load MOG results from {mog_path}")
        return

    # Load all baseline results first
    baseline_detailed_results = {}
    all_metrics_results = {}

    for baseline_name, baseline_path in baseline_paths.items():
        baseline_detailed = load_metrics_from_json(baseline_path)
        if not baseline_detailed:
            logging.error(
                f"Failed to load {baseline_name} results from {baseline_path}"
            )
            continue

        baseline_detailed_results[baseline_name] = baseline_detailed

        # Run analysis for each baseline to get metrics
        metrics_results = analyze_all_metrics(mog_detailed, baseline_detailed)
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

    print("\n=== MOG vs Best Baseline per Metric ===")

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
    save_path = os.path.join(output_dir, "ngram_pvalue.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    logging.info(f"P-value analysis completed for MOG vs best baselines per metric.")
    logging.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute p-values between MOG and the best baseline for each metric"
    )
    parser.add_argument(
        "--mog_results", required=True, help="Path to MOG detailed results JSON file"
    )
    parser.add_argument(
        "--baseline_results",
        required=True,
        nargs="+",
        help="Paths to baseline results JSON files in format 'name:path'",
    )
    parser.add_argument(
        "--output_dir", default="pvalue_results", help="Directory to save result files"
    )

    args = parser.parse_args()

    # Parse baseline paths
    baseline_paths = {}
    for baseline_arg in args.baseline_results:
        name, path = baseline_arg.split(":", 1)
        baseline_paths[name] = path

    compare_with_best_baselines_per_metric(
        args.mog_results, baseline_paths, args.output_dir
    )
