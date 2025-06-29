import os
import argparse
import sys

sys.path.append("./src")
sys.path.append(".")

from agents.planner_without_mo import Planner
from lm import OpenAILM
from utils import Parser
from config import Config


def setup_language_model(model_name: str, max_tokens: int):
    """Create and return the language model we'll use."""
    common_params = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_BASE_URL"],
        "max_tokens": max_tokens,
        "support_structured_response": True,
    }

    return OpenAILM(model=model_name, **common_params)


def process_topic(
    topic, lm, better_lm, config, skip_research=False, skip_outline=False
):
    """Generate an article for a given topic."""
    planner = Planner(
        topic=topic.strip(), engine=lm, better_engine=better_lm, config=config
    )

    return planner.execute(skip_research=skip_research, skip_outline=skip_outline)


def get_topics(file_path=None):
    """Get list of topics from either file or console input."""
    if file_path and os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.readlines()

    topic = input("Enter a topic to generate an article about: ").strip()
    return [topic] if topic else []


def main(file_path=None, skip_research=False, skip_outline=False):
    # Initialize configuration and models
    config = Config()
    lm = setup_language_model(config.MODEL_NAME, config.MAX_TOKENS)
    better_lm = setup_language_model(config.BETTER_MODEL_NAME, config.MAX_TOKENS)

    # Process each topic
    articles = []
    for topic in get_topics(file_path):
        output_filename = Parser.safe_title(topic.strip())
        topic_path = os.path.join(
            config.OUTPUT_DIR, "raw_txt", f"{output_filename}.txt"
        )

        if os.path.exists(topic_path):
            print(f"Skipping {topic.strip()} - output file already exists")
            continue

        try:
            article = process_topic(
                topic,
                lm,
                better_lm,
                config,
                skip_research=skip_research,
                skip_outline=skip_outline,
            )
            articles.append(article)
            print(f"Successfully generated article for: {topic.strip()}")
        except Exception as e:
            print(f"Error processing {topic.strip()}: {e}")

    return articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Optional: Path to file containing topics (one per line)",
    )
    parser.add_argument(
        "--skip-research", action="store_true", help="Skip the research phase"
    )
    parser.add_argument(
        "--skip-outline", action="store_true", help="Skip the outline generation phase"
    )

    args = parser.parse_args()
    main(
        args.file_path, skip_research=args.skip_research, skip_outline=args.skip_outline
    )
