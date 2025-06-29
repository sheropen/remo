# Acknowledgement: This part of code is adapted from https://github.com/stanford-oval/storm.
import json
import numpy as np
import os
import sys
import dspy
import re
from tqdm import tqdm
import concurrent.futures
import argparse

sys.path.append("./src")
sys.path.append(".")
from config import Config
from article import Article
from lm import OpenAILM


class Entailer(dspy.Signature):
    """Is the claim faithful to the source? A claim is faithful to the source if the core part in the claim can be supported by the source.
    Start your answer with 'Yes' or 'No'."""

    source = dspy.InputField()
    claim = dspy.InputField()
    answer = dspy.OutputField(desc="reply only 'Yes' or 'No'")


class PartialEntailer(dspy.Signature):
    """Can the source at least partially support the claim?
    Start your answer with 'Yes' or 'No'.
    """

    source = dspy.InputField()
    claim = dspy.InputField()
    answer = dspy.OutputField(desc="reply only 'Yes' or 'No'")


class NLI(dspy.Module):
    def __init__(self):
        super().__init__()
        self.entail = dspy.ChainOfThought(Entailer)
        self.partial_entail = dspy.ChainOfThought(PartialEntailer)

    def forward(self, source, claim, partial=False):
        if partial:
            answer = self.partial_entail(source=source, claim=claim)
        else:
            answer = self.entail(source=source, claim=claim)
        return answer


def extract_sentence_cite_pairs(article, use_proposition=True):
    sentence_cite_pairs = []
    if article.layer != 0:
        for paragraph in article.paragraph_list:
            for sentence in paragraph.sentence_list:
                if sentence.proposition is None or use_proposition == False:
                    sentence_cite_pairs.append(
                        (sentence.content, sentence.citation_list)
                    )
                else:
                    sentence_cite_pairs.append(
                        (sentence.proposition, sentence.citation_list)
                    )
    for subsection in article.subsection_list:
        sentence_cite_pairs.extend(extract_sentence_cite_pairs(subsection))
    return sentence_cite_pairs


def truncate_paragraph(paragraph, max_words):
    # Tokenize paragraph into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", paragraph)

    # Tokenize each sentence into words and form trunks
    trunks = []
    current_trunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()  # Tokenize sentence into words
        sentence_word_count = len(sentence_words)

        if current_word_count + sentence_word_count <= max_words:
            current_trunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            trunks.append(" ".join(current_trunk))
            current_trunk = [sentence]
            current_word_count = sentence_word_count

    if current_trunk:
        trunks.append(" ".join(current_trunk))

    return trunks


def remove_citations(sent):
    return (
        re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent))
        .replace(" |", "")
        .replace("]", "")
    )


def _run_nli(premise, hypothesis, partial=False):
    lm = OpenAILM(
        model="gpt-4o-mini-2024-07-18",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_tokens=4000,
        support_structured_response=True,
    )
    dspy.settings.configure(lm=lm)
    check_citation = NLI()
    answer = (
        check_citation(source=premise, claim=hypothesis, partial=partial)
        .answer.strip()
        .lower()
    )
    if "yes" in answer:
        return True
    elif "no" in answer:
        return False
    else:
        print(f"Invalid answer: {answer}")
        return False


def process_sentence(sent, citations, at_most_citations):
    target_sent = remove_citations(sent).strip()

    entail = 0
    entail_prec = 0
    total_citations = 0
    joint_entail = -1  # Undecided
    unnecessary_citations = []

    # Find references
    if citations is None or len(citations) == 0:
        joint_entail = 0
    else:
        if at_most_citations is not None:
            citations = citations[:at_most_citations]
        total_citations = len(citations)
        joint_passage = "\n".join([citation.content for citation in citations])

        # If not directly rejected by citation format error, calculate the recall score
        if joint_entail == -1:
            joint_entail = _run_nli(joint_passage, target_sent, partial=False)

    entail += joint_entail
    if joint_entail == 0:
        print(f"[Unsupported sentence] {sent}")

    # calculate the precision score if applicable
    if joint_entail and len(citations) > 1:
        # Precision check: did the model cite any unnecessary documents?
        for ref_idx, citation in enumerate(citations):
            passage = citation.content
            support_result = _run_nli(passage, target_sent, partial=True)
            if not support_result:
                print(f"[Unnecessary citation] sent: {sent} citation: [{citation.url}]")
                unnecessary_citations.append(citation.url)
            else:
                entail_prec += 1
    else:
        entail_prec += joint_entail

    return {
        "sent": sent,
        "target_sent": target_sent,
        "citation_list": [citation.to_dict() for citation in citations],
        "joint_entail": joint_entail,
        "unnecessary_citations": unnecessary_citations,
        "entail": entail,
        "entail_prec": entail_prec,
        "total_citations": total_citations,
    }


def compute_autoais(
    data, decontext=False, concat=False, qampari=False, at_most_citations=None
):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    ais_scores = []
    ais_scores_prec = []

    sent_uncitated = 0
    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0

    eval_log = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_sent = {
            executor.submit(process_sentence, sent, citations, at_most_citations): (
                sent,
                citations,
            )
            for sent, citations in data
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sent), total=len(data)
        ):
            sent, citations = future_to_sent[future]
            try:
                result = future.result()
                eval_log.append(result)
                sent_total += 1
                ais_scores.append(result["entail"])
                ais_scores_prec.append(
                    result["entail_prec"] / result["total_citations"]
                    if result["total_citations"] > 0
                    else 0
                )

                if result["total_citations"] == 0:
                    sent_uncitated += 1
                if len(citations) > 1:
                    sent_mcite += 1
                    if result["joint_entail"]:
                        sent_mcite_support += 1
                        if result["unnecessary_citations"]:
                            sent_mcite_overcite += 1
            except Exception as exc:
                print(f"Sentence {sent} generated an exception: {exc}")

    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support,
            )
        )

    citation_rec = 100 * np.mean(ais_scores)
    citation_prec = 100 * np.mean(ais_scores_prec)
    sent_uncitated_rate = 100 * sent_uncitated / sent_total

    return {
        "evaluation_logs": eval_log,
        "citation_rec": citation_rec,
        "citation_prec": citation_prec,
        "sent_uncitated_rate": sent_uncitated_rate,
    }


def process_folder(folder_path, use_proposition=True):
    config = Config()

    # Create eval/citation directory if it doesn't exist
    output_folder_path = os.path.join(folder_path, "eval", "citation")
    os.makedirs(output_folder_path, exist_ok=True)

    results = {}
    # First load any existing results
    results_file = os.path.join(output_folder_path, "citation_eval_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        # Skip if we already have results for this file
        if filename in results:
            print(f"Skipping {filename} - already processed")
            continue

        # Check if individual result file exists
        individual_result_file = os.path.join(
            output_folder_path, f"{filename}_result.json"
        )
        if os.path.exists(individual_result_file):
            print(f"Loading cached result for {filename}")
            with open(individual_result_file, "r") as f:
                eval_result = json.load(f)
            results[filename] = eval_result
            continue

        try:
            article_path = os.path.join(folder_path, filename)
            with open(article_path, "r") as f:
                article_json = json.load(f)

            article = Article.from_dict(article_json)
            sentence_cite_pairs = extract_sentence_cite_pairs(article, use_proposition)

            eval_result = compute_autoais(sentence_cite_pairs)

            # Save individual result
            with open(individual_result_file, "w") as f:
                json.dump(eval_result, f, indent=2)

            results[filename] = eval_result

            # Update the main results file after each successful evaluation
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Processed {filename}")
            print(eval_result)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process folder for citation evaluation"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="/home/junhao/projects/oreo/eval/storm",
        help="Path to the folder containing JSON files",
    )
    parser.add_argument(
        "--proposition",
        action="store_true",
        default=False,
        help="Use proposition instead of sentence for evaluation",
    )
    args = parser.parse_args()

    folder_path = args.folder

    if args.proposition:
        print("Using propositions for evaluation")
    else:
        print("Using sentences for evaluation")

    results = process_folder(folder_path, args.proposition)

    output_folder_path = os.path.join(folder_path, "eval", "citation")
    os.makedirs(output_folder_path, exist_ok=True)
    output_file = os.path.join(output_folder_path, "citation_eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

    # Average result
    avg_citation_rec = sum(
        [result["citation_rec"] for result in results.values()]
    ) / len(results)
    avg_citation_prec = sum(
        [result["citation_prec"] for result in results.values()]
    ) / len(results)
    avg_sent_uncitated_rate = sum(
        [result["sent_uncitated_rate"] for result in results.values()]
    ) / len(results)
    print(f"Average citation rec: {avg_citation_rec}")
    print(f"Average citation prec: {avg_citation_prec}")
    print(f"Average sent uncitated rate: {avg_sent_uncitated_rate}")
