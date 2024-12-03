import os
import argparse
from collections import defaultdict
import spacy
from spacy.cli import download
import json
import re

# Initialize spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading English model...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def analyze_text(text):
    """Analyze a text segment for word count, entities, and special numbers."""
    doc = nlp(text)
    # Find special numbers: 4-digit numbers, percentage, decimal numbers, date
    special_numbers = set(
        re.findall(r"\b(\d{4}|\d+%|\d+\.\d+|\d{1,2}/\d{1,2}/\d{2,4})\b", text)
    )
    return {
        "word_count": len(text.split()),
        "entities": {ent.text for ent in doc.ents},
        "special_numbers": special_numbers,
    }


def collect_text_stats(text):
    """Collect statistics from a text document."""
    stats = defaultdict(int)
    all_entities = set()
    all_numbers = set()

    # Split text into sections (assuming sections start with #)
    sections = [s.strip() for s in text.split("\n#") if s.strip()]
    stats["section_count"] = len(sections)

    # Process each section
    for section in sections:
        # Split into paragraphs (assuming paragraphs are separated by blank lines)
        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        stats["paragraph_count"] += len(paragraphs)

        for paragraph in paragraphs:
            # Split into sentences (using spacy)
            doc = nlp(paragraph)
            sentences = list(doc.sents)
            stats["sentence_count"] += len(sentences)

            for sentence in sentences:
                text_analysis = analyze_text(str(sentence))
                stats["word_count"] += text_analysis["word_count"]
                all_entities.update(text_analysis["entities"])
                all_numbers.update(text_analysis["special_numbers"])

    # Add unique counts
    stats["unique_entity_count"] = len(all_entities)
    stats["special_number_count"] = len(all_numbers)

    return stats


def process_folder(folder_path):
    """Process all text files in a folder and return statistics."""
    results = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            stats = collect_text_stats(text)
            results[filename] = stats
            print(f"Processed {filename}")
            print(stats)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    return results


def calculate_averages(results):
    """Calculate average statistics across all articles."""
    totals = defaultdict(int)
    for stats in results.values():
        for key, value in stats.items():
            totals[key] += value

    return {key: value / len(results) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Process folder for text statistics")
    parser.add_argument(
        "--folder",
        type=str,
        default="/home/junhao/projects/oreo/data/output/fresh_wiki/mog/clean_txt",
        help="Path to the folder containing text files",
    )
    args = parser.parse_args()

    # Process articles and save results
    results = process_folder(args.folder)
    output_folder = os.path.join(args.folder, "eval")
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "text_statistics.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Print statistics
    avg_stats = calculate_averages(results)
    print("\nAverage Statistics:")
    for key, value in avg_stats.items():
        print(f"Average {key}: {value:.2f}")
    with open(os.path.join(output_folder, "text_statistics_avg.txt"), "w") as f:
        for key, value in avg_stats.items():
            f.write(f"{key}: {value:.2f}\n")


if __name__ == "__main__":
    main()
