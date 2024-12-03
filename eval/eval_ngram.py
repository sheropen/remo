import os
import sys
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict, Union
import logging
import json
from tqdm import tqdm
import spacy
import re
from collections import defaultdict

sys.path.append("./src")

from utils import Parser

# Initialize spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading English model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def load_text(filepath: str) -> str:
    """Load text content from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {e}")
        return ""


def analyze_text_stats(text: str) -> Dict[str, set]:
    """Extract entities and special numbers from text."""
    doc = nlp(text)
    special_numbers = set(
        re.findall(r"\b(\d{4}|\d+%|\d+\.\d+|\d{1,2}/\d{1,2}/\d{2,4})\b", text)
    )
    entities = {ent.text.lower() for ent in doc.ents}

    return {"entities": entities, "special_numbers": special_numbers}


def evaluate_text_pair(reference: str, candidate: str) -> Dict[str, float]:
    """
    Evaluate a candidate text against a reference using various metrics.
    """
    metrics = {}

    # Tokenize texts
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())

    # Calculate BLEU scores
    smoothing = SmoothingFunction().method1
    for n in range(1, 5):
        metrics[f"bleu-{n}"] = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=tuple([1.0 / n] * n + [0] * (4 - n)),
            smoothing_function=smoothing,
        )

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    for metric, score in rouge_scores.items():
        metrics[f"{metric}_precision"] = score.precision
        metrics[f"{metric}_recall"] = score.recall
        metrics[f"{metric}_fmeasure"] = score.fmeasure

    # Calculate entity and special number recall
    ref_stats = analyze_text_stats(reference)
    cand_stats = analyze_text_stats(candidate)

    # Entity recall
    if ref_stats["entities"]:
        shared_entities = ref_stats["entities"].intersection(cand_stats["entities"])
        metrics["entity_recall"] = len(shared_entities) / len(ref_stats["entities"])
    else:
        metrics["entity_recall"] = 1.0  # Perfect recall if no entities in reference

    # Special number recall
    if ref_stats["special_numbers"]:
        shared_numbers = ref_stats["special_numbers"].intersection(
            cand_stats["special_numbers"]
        )
        metrics["special_number_recall"] = len(shared_numbers) / len(
            ref_stats["special_numbers"]
        )
    else:
        metrics["special_number_recall"] = (
            1.0  # Perfect recall if no special numbers in reference
        )

    return metrics


def evaluate_folder_pair(
    ref_folder: str, cand_folder: str, topic_list: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all texts in candidate folder against references using topic list.

    Args:
        ref_folder: Path to reference texts folder
        cand_folder: Path to candidate texts folder
        topic_list: Path to file containing list of topics

    Returns:
        Dictionary of topics and their metric scores
    """
    results = {}

    # Create eval directory if it doesn't exist
    eval_dir = os.path.join(cand_folder, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Load topic list
    with open(topic_list, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]

    topics = [Parser.safe_title(topic) for topic in topics]
    for topic in tqdm(topics, desc="Evaluating topics"):
        ref_path = os.path.join(ref_folder, f"{topic}.txt")
        cand_path = os.path.join(cand_folder, f"{topic}.txt")

        if not os.path.exists(ref_path) or not os.path.exists(cand_path):
            logging.warning(f"Missing files for topic {topic}")
            continue

        ref_text = load_text(ref_path)
        cand_text = load_text(cand_path)

        if not ref_text or not cand_text:
            continue

        results[topic] = evaluate_text_pair(ref_text, cand_text)

    return results


def compute_average_scores(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute average scores across all topics."""
    if not results:
        return {}

    all_metrics = list(next(iter(results.values())).keys())
    averages = {}

    for metric in all_metrics:
        scores = [topic_results[metric] for topic_results in results.values()]
        averages[metric] = sum(scores) / len(scores)

    return averages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate n-gram metrics between reference and candidate texts"
    )
    parser.add_argument(
        "--ref_folder", required=True, help="Path to reference texts folder"
    )
    parser.add_argument(
        "--cand_folder", required=True, help="Path to candidate texts folder"
    )
    parser.add_argument("--topic_list", required=True, help="Path to topic list file")

    args = parser.parse_args()

    # Evaluate all texts
    results = evaluate_folder_pair(args.ref_folder, args.cand_folder, args.topic_list)

    # Compute average scores
    averages = compute_average_scores(results)

    # Save detailed results
    detailed_path = os.path.join(args.cand_folder, "eval", "ngram_detailed.json")
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save averaged results
    averaged_path = os.path.join(args.cand_folder, "eval", "ngram_averaged.json")
    with open(averaged_path, "w", encoding="utf-8") as f:
        json.dump(averages, f, indent=2)

    print("\nResults saved to:")
    print(f"Detailed results: {detailed_path}")
    print(f"Averaged results: {averaged_path}")

    print("\nAverage scores across all topics:")
    for metric, score in averages.items():
        print(f"{metric}: {score:.4f}")
