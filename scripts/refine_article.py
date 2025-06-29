import sys
from pathlib import Path
import dspy
import os
import argparse
from tqdm import tqdm

# Add project path to system path
sys.path.append("./src")
sys.path.append(".")
from article import Article
from utils import Parser
from config import Config
from lm import OpenAILM


def setup_language_model(model_name: str, max_tokens: int):
    """Create and return the language model we'll use."""
    common_params = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_BASE_URL"],
        "max_tokens": max_tokens,
        "support_structured_response": True,
    }
    return OpenAILM(model=model_name, **common_params)


class Refiner(dspy.Signature):
    """
    Refine the section of the given topic while keeping the markdown section headings unchanged.
    Stay focus on the section, remove unnecessary information about the section, and improve the coherence and writing.
    Only output the refined text, no other information.
    """

    topic = dspy.InputField(desc="Topic of the text")
    text = dspy.InputField(desc="Text to refine")
    refined_text = dspy.OutputField(desc="Refined text")


refine_section = dspy.Predict(Refiner)


def get_working_context_dict(section, parent_path=""):
    """Recursively build working context dictionary for a section and its subsections."""
    context_dict = {}

    # Build full section path
    full_path = f"{parent_path}//{section.title}" if parent_path else section.title

    # Add section's working context to dict
    context_dict[full_path] = section.working_context

    # Recursively process subsections
    for subsection in section.subsection_list:
        context_dict.update(get_working_context_dict(subsection, full_path))

    return context_dict


def copy_working_contexts(refined_section, working_context_dict, parent_path=""):
    """Copy working contexts from original to refined section."""
    # Build full section path
    full_path = (
        f"{parent_path}//{refined_section.title}"
        if parent_path
        else refined_section.title
    )

    # Copy working context if it exists in original dict
    if full_path in working_context_dict:
        refined_section.working_context = working_context_dict[full_path]

    # Recursively process subsections
    for refined_subsec in refined_section.subsection_list:
        copy_working_contexts(refined_subsec, working_context_dict, full_path)


def verify_section_headings(original_section, refined_section):
    """Verify that section headings remain unchanged after refinement."""
    # Check the main section title
    if original_section.title != refined_section.title:
        raise ValueError(
            f"Section title mismatch: {original_section.title} != {refined_section.title}"
        )

    # Check subsection titles
    orig_subsections = {s.title for s in original_section.subsection_list}
    refined_subsections = {s.title for s in refined_section.subsection_list}

    if orig_subsections != refined_subsections:
        missing = orig_subsections - refined_subsections
        extra = refined_subsections - orig_subsections
        error_msg = []
        if missing:
            error_msg.append(f"Missing sections: {missing}")
        if extra:
            error_msg.append(f"Extra sections: {extra}")
        raise ValueError(" | ".join(error_msg))


def refine_article(article):
    """Refine an entire article section by section."""
    refined_article = Article(title=article.title)
    failed_count = 0
    total_sections = len(article.subsection_list)

    for section in article.subsection_list:
        try:
            # Get working contexts before refinement
            working_context_dict = get_working_context_dict(section)

            # Refine the section
            section_text = f"{section}"
            refine_section_text = refine_section(
                topic=article.title, text=section_text
            ).refined_text

            # Create refined section
            refined_section = article.from_text(
                title=section.title, text=refine_section_text, layer=1
            )

            # Verify section headings
            verify_section_headings(section, refined_section)

            # Copy working contexts to refined section
            copy_working_contexts(refined_section, working_context_dict)

            # Add refined section to article
            refined_article.subsection_list.append(refined_section)

        except ValueError as e:
            print(f"Section heading verification failed for {section.title}: {str(e)}")
            failed_count += 1
            refined_article.subsection_list.append(section)
            continue

    failure_rate = (failed_count / total_sections) * 100 if total_sections > 0 else 0
    print(
        f"Refinement failed for {failed_count}/{total_sections} sections ({failure_rate:.1f}%)"
    )

    return refined_article, failed_count


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Refine articles by improving coherence and writing")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing JSON files to refine"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for refined articles"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    output_dir.mkdir(exist_ok=True)

    # Setup language model
    config = Config()
    lm = setup_language_model("gpt-4o-2024-08-06", 4000)
    dspy.configure(lm=lm)

    # Process all articles
    article_files = list(input_dir.glob("*.json"))
    total_failed = 0
    total_sections = 0

    for article_file in tqdm(article_files, desc="Processing articles"):
        try:
            article = Article.from_json(str(article_file))
            output_path = os.path.join(
                str(output_dir),
                "clean_txt",
                f"{Parser.safe_title(article.title)}.txt",
            )
            if os.path.exists(output_path):
                print(f"Skipping {article_file} because it already exists")
                continue
            total_sections += len(article.subsection_list)

            # Get both refined article and failure count
            refined_article, failed = refine_article(article)
            total_failed += failed

            refined_article.save_file(str(output_dir))

        except Exception as e:
            print(f"Error processing {article_file}: {str(e)}")

    # Print overall statistics
    overall_failure_rate = (
        (total_failed / total_sections) * 100 if total_sections > 0 else 0
    )
    print(f"\nOverall refinement statistics:")
    print(
        f"Total failed sections: {total_failed}/{total_sections} ({overall_failure_rate:.1f}%)"
    )


if __name__ == "__main__":
    main()
