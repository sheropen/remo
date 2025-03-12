import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from deep_research import main as process_topic
from src.utils import setup_logger

logger = setup_logger()


def read_topics(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_completed_topics():
    completed_file = Path("data/topic/all_topics.txt")
    if not completed_file.exists():
        completed_file.parent.mkdir(parents=True, exist_ok=True)
        completed_file.touch()
        return set()

    with open(completed_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def add_completed_topic(topic):
    with open("data/topic/all_topics.txt", "a", encoding="utf-8") as f:
        f.write(f"{topic}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple topics for deep research"
    )
    parser.add_argument(
        "topics_file", type=str, help="Path to file containing topics (one per line)"
    )
    parser.add_argument(
        "--skip-research", action="store_true", help="Skip research phase"
    )
    parser.add_argument(
        "--skip-outline", action="store_true", help="Skip outline generation"
    )
    parser.add_argument("--skip-write", action="store_true", help="Skip write phase")
    parser.add_argument(
        "--force-recreate", action="store_true", help="Force recreate memory"
    )

    args = parser.parse_args()

    topics = read_topics(args.topics_file)
    completed_topics = read_completed_topics()
    topics_to_process = [t for t in topics if t not in completed_topics]

    logger.info(f"Found {len(topics)} topics, {len(topics_to_process)} need processing")
    failed_topics = []

    for topic in tqdm(topics_to_process, desc="Processing topics"):
        try:
            logger.info(f"Starting to process topic: {topic}")
            process_topic(
                topic,
                skip_research=args.skip_research,
                skip_outline=args.skip_outline,
                skip_write=args.skip_write,
                force_recreate=args.force_recreate,
            )
            add_completed_topic(topic)
            logger.info(f"Successfully processed topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to process topic '{topic}': {str(e)}")
            failed_topics.append((topic, str(e)))
            continue

    if failed_topics:
        logger.error("\nFailed topics:")
        for topic, error in failed_topics:
            logger.error(f"- {topic}: {error}")

    logger.info(
        f"\nProcessing completed. Successfully processed {len(topics) - len(failed_topics)}/{len(topics)} topics."
    )


if __name__ == "__main__":
    main()
