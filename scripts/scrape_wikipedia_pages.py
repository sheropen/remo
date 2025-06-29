import sys
import logging
from pathlib import Path
import argparse
import wikipediaapi

sys.path.append("./src")
from article import Article
from utils import Parser

# Constants
EXCLUDED_SECTIONS = {
    "introduction",
    "overview",
    "see also",
    "references",
    "external links",
    "further reading",
    "footnotes",
    "notes",
    "bibliography",
    "citations",
    "gallery",
    "sources",
    "additional information",
    "supplementary materials",
    "conclusion",
    "appendix",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("wikipedia_scraping.log"), logging.StreamHandler()],
)


def get_section(section, layer: int = 1) -> Article:
    """Recursively process article sections."""
    article = Article.from_text(title=section.title, text=section.text, layer=layer)
    for subsection in section.sections:
        article.add_subsection(get_section(subsection, layer + 1))
    return article


def process_article(title: str, output_dir: Path) -> bool:
    """Process a single Wikipedia article and save it to JSON."""
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent="Knowledge Curation Project",
            language="en",
        )
        page = wiki.page(title)

        # Create main article with summary
        article = Article.from_text(title=page.title, text=page.summary, layer=0)

        # Process sections
        for section in page.sections:
            if section.title.lower() not in EXCLUDED_SECTIONS:
                article.add_subsection(get_section(section))

        # Save article
        article.save_file(str(output_dir))
        logging.info(f"Successfully saved article: {title}")
        return True

    except Exception as e:
        logging.error(f"Failed to process article '{title}': {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="data/topic_list.txt",
        help="Input file containing topic list",
    )
    parser.add_argument(
        "-o", "--output", default="dataset", help="Output directory for JSON files"
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read and process topics
    try:
        with open(args.input, "r") as f:
            topics = f.read().splitlines()
    except FileNotFoundError:
        logging.error(f"Input file {args.input} not found")
        return

    if not topics:
        logging.warning("No topics found in input file")
        return

    # Process articles
    results = [process_article(title, output_dir) for title in topics]
    processed = sum(results)
    failed = len(results) - processed

    logging.info(f"Scraping completed. Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    main()
