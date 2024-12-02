import argparse
import random
from pathlib import Path


def split_topic_list(input_file: str, test_size: int = 100, seed: int = 42):
    """
    Split a topic list file into training and test sets.

    Args:
        input_file (str): Path to the input topic list file
        test_size (int): Number of topics for test set (default: 100)
        seed (int): Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Read topics from file and filter out "list of" topics
    with open(input_file, "r", encoding="utf-8") as f:
        topics = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().lower().startswith("list of")
        ]

    if len(topics) <= test_size:
        raise ValueError(
            f"Not enough topics ({len(topics)}) for test set size ({test_size})"
        )

    # Shuffle topics
    random.shuffle(topics)

    # Split into train and test sets
    test_topics = topics[:test_size]
    train_topics = topics[test_size:]

    # Create output file paths
    input_path = Path(input_file)
    output_dir = input_path.parent
    base_name = input_path.stem

    train_file = output_dir / f"{base_name}_train.txt"
    test_file = output_dir / f"{base_name}_test.txt"

    # Write train set
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_topics))

    # Write test set
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_topics))

    print(f"Total topics (excluding 'list of'): {len(topics)}")
    print(f"Training topics: {len(train_topics)}")
    print(f"Test topics: {len(test_topics)}")
    print(f"Training set saved to: {train_file}")
    print(f"Test set saved to: {test_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Split topic list into train and test sets"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input topic list file"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of topics for test set (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    split_topic_list(args.input_file, args.test_size, args.seed)


if __name__ == "__main__":
    main()
