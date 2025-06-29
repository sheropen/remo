import sys
from pathlib import Path
import dspy
import os
from tqdm import tqdm

# Add project path to system path
sys.path.append("/home/junhao/projects/mog/src")
from article import Article
from utils import Config, Parser
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


class CitationFinder(dspy.Signature):
    """Given a sentence and a source list, only cite the most relevant sources to fully cover the sentence.
    The output should be a list of indices of the sources in the input list.
    """

    claim = dspy.InputField()
    source_list = dspy.InputField(desc="a list of source text")
    answer = dspy.OutputField(
        type=list[int],
        desc="the indices of the most relevant source text, reply only the indices",
    )


def find_citations_for_section(
    section, citation_finder, lm, better_lm, root_article: Article
):
    """Find citations for all sentences in a section."""
    total_sentences = 0
    uncited_sentences = 0

    # Process current section's paragraphs
    for paragraph in section.paragraph_list:
        for sentence in paragraph.sentence_list:
            total_sentences += 1
            if not sentence.citation_list:
                sentence.citation_list = []
                # Create text with numbered sources
                mu_text = ""
                for idx, memory_unit in enumerate(section.working_context):
                    mu_text += f"{idx+1}: {memory_unit.content}\n"

                # Find relevant source indices
                with dspy.context(lm=lm):
                    try:
                        answer = citation_finder(
                            claim=sentence.content, source_list=mu_text
                        )
                        indices = [int(idx) for idx in answer.answer.split(", ")]
                        for idx in indices:
                            if 0 <= idx - 1 < len(section.working_context):
                                sentence.citation_list.append(
                                    section.working_context[idx - 1]
                                )
                    except Exception as e:
                        try:
                            with dspy.context(lm=better_lm):
                                answer = citation_finder(
                                    claim=sentence.content, source_list=mu_text
                                )
                                indices = [
                                    int(idx) for idx in answer.answer.split(", ")
                                ]
                                for idx in indices:
                                    if 0 <= idx - 1 < len(section.working_context):
                                        sentence.citation_list.append(
                                            section.working_context[idx - 1]
                                        )
                        except Exception as e:
                            print(
                                f"Error processing sentence: {sentence.content}\nError: {e}"
                            )
                            uncited_sentences += 1
                        continue

            sentence.doc_id_list = []
            for mu in sentence.citation_list:
                doc_id = root_article.record_cited_mu(mu)
                sentence.doc_id_list.append(doc_id)

    # Recursively process subsections
    for subsection in section.subsection_list:
        sub_total, sub_uncited = find_citations_for_section(
            subsection, citation_finder, lm, better_lm, root_article
        )
        total_sentences += sub_total
        uncited_sentences += sub_uncited

    return total_sentences, uncited_sentences


def process_article(article, citation_finder, lm, better_lm):
    """Process all sections in an article to find citations."""
    total_sentences, uncited_sentences = find_citations_for_section(
        article, citation_finder, lm, better_lm, article
    )
    citation_ratio = (
        1 - (uncited_sentences / total_sentences) if total_sentences > 0 else 0
    )
    print(
        f"Citation ratio: {citation_ratio:.2%} ({total_sentences - uncited_sentences}/{total_sentences} sentences cited)"
    )
    return article


def main():
    config = Config()

    # Setup paths
    input_dir = Path(
        "/home/junhao/projects/mog/data/output/mix_wiki/mog_wo_mo-p/json"
    )
    output_dir = Path(
        "/home/junhao/projects/mog/data/output/mix_wiki/mog_wo_mo-p-cited"
    )
    output_dir.mkdir(exist_ok=True)

    # Setup models
    lm = setup_language_model("gpt-4o-mini-2024-07-18", 4000)
    better_lm = setup_language_model("gpt-4o-2024-08-06", 4000)
    citation_finder = dspy.Predict(CitationFinder)

    # Track total statistics
    total_sentences_all = 0
    uncited_sentences_all = 0

    # Process all articles
    article_files = list(input_dir.glob("*.json"))
    for article_file in tqdm(article_files, desc="Processing articles"):
        try:
            # Load article
            article = Article.from_json(article_file)

            # Process article and get statistics
            total_sentences, uncited_sentences = find_citations_for_section(
                article, citation_finder, lm, better_lm, article
            )
            total_sentences_all += total_sentences
            uncited_sentences_all += uncited_sentences

            # Save processed article
            article.save_file(output_dir)

            # Print individual article statistics
            citation_ratio = (
                1 - (uncited_sentences / total_sentences) if total_sentences > 0 else 0
            )
            print(
                f"Article {article_file.name} - Citation ratio: {citation_ratio:.2%} ({total_sentences - uncited_sentences}/{total_sentences} sentences cited)"
            )

        except Exception as e:
            print(f"Error processing {article_file.name}: {e}")
            continue

    # Print overall statistics
    overall_ratio = (
        1 - (uncited_sentences_all / total_sentences_all)
        if total_sentences_all > 0
        else 0
    )
    print("\nOverall Statistics:")
    print(
        f"Total citation ratio: {overall_ratio:.2%} ({total_sentences_all - uncited_sentences_all}/{total_sentences_all} sentences cited)"
    )


if __name__ == "__main__":
    main()
