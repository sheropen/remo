import argparse
import copy
import glob
import json
import logging
import os
import re

from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer, LlamaForCausalLM
from evaluation_trim_length import process_document

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_txt_file(file_path):
    """
    Read a text file to string
    """
    with open(file_path, "r") as file:
        return file.read()


def read_json(file_path):
    """
    Read a json file to dict
    """
    with open(file_path, "r") as file:
        return json.load(file)


def preprocess_text(text):
    """
    Clean up text: remove reference section, URLS, non-ascii chars
    """
    # clean up empty line
    paragraphs = text.split("\n")
    paragraphs = [i for i in paragraphs if len(i) > 0]
    # clean up section title and remove reference section
    cleaned_pargraphs = []
    for i in paragraphs:
        if i == "# References":
            break
        if i.startswith("#"):
            i = "section: " + i.replace("#", "").strip()
        cleaned_pargraphs.append(i)
    text = "\n".join(cleaned_pargraphs)
    # remove URLS
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # remove non-ascii char
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    # remove citation bracket (e.g. [10])
    text = re.sub(r"\[\d+\]", "", text)
    # remove non alphanumeric char
    text = re.sub(r"[^\w\s]", "", text)
    return text


def get_conversation_prompt(filled_prompt):
    """
    From filled prompt, convert it into llama-2 conversation prompt
    """
    conv = get_conv_template("llama-2")
    conv.set_system_message("You are a fair evaluator language model.")
    conv.append_message(conv.roles[0], filled_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def format_prompt(prompt_template, topic, rubric, response):
    """
    Fill prompt_template with rubric and response
    """
    prompt_template_copy = copy.deepcopy(prompt_template)
    data = copy.deepcopy(rubric)
    data.update(
        {
            "instruction": f"You are a Wikipedia editor. Your task is write a wikipedia page for the topic: {topic}",
            "response": response,
        }
    )
    filled_prompt = prompt_template_copy.format(**data)
    return get_conversation_prompt(filled_prompt)


def get_grading_dict(
    responses,
    topic,
    tokenizer,
    model,
    prompt_template_path="./eval_prometheus_no_ref.prompt",
    rubric_path="./eval_rubric_5.json",
    disable_sample=False,
    temperature=0.01,
    top_p=0.95,
    max_new_tokens=512,
    repetition_penalty=1.03,
    logger=None,
):
    grading = {}
    prompt_template = read_txt_file(prompt_template_path)
    rubrics = read_json(rubric_path)

    # Read all files in the given directory
    for rubric_idx, rubric in enumerate(rubrics):
        grading[rubric["criteria_description"]] = {}
        for response_idx, response in enumerate(responses):
            # generate evaluation prompt and tokenize
            if logger is not None:
                logger.info(
                    f"processing for rubric {rubric_idx + 1}/{len(rubrics)}, response {response_idx + 1}/{len(responses)}, response length: {len(response)}"
                )
            prompt = format_prompt(
                prompt_template=prompt_template,
                topic=topic,
                rubric=rubric,
                response=response,
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            # geenrate output
            outputs = model.generate(
                input_ids,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=not disable_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            decoded_output = tokenizer.decode(outputs[0])
            # decode output and format into desired fields
            decoded_output = decoded_output[
                decoded_output.find("[/INST]") + len("[/INST]") :
            ].strip()
            feedback = decoded_output[: decoded_output.find("[RESULT]")]
            score = (
                decoded_output[decoded_output.find("[RESULT]") + len("[RESULT]") :]
                .replace("</s>", "")
                .strip()
            )
            try:
                int(score)
            except Exception as e:
                pattern = r"the overall score is (\d+)"
                match = re.search(pattern, feedback)
                if match:
                    score = match.group(1)

            grading[rubric["criteria_description"]][response_idx] = {
                "feedback": feedback,
                "score": score,
            }
    return grading


def read_topic_list(file_path):
    """
    Read topics from a file, one per line
    """
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def get_topics_from_directory(input_dir):
    """
    Get topics from all .txt files in the directory
    """
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    return [os.path.splitext(os.path.basename(f))[0] for f in txt_files]


def load_model(args):
    """
    Load model and tokenizer with flexible path handling:
    - If model_path provided: load from {model_path}/{model}
    - If no model_path: load directly from HuggingFace hub
    """
    logger.info("Loading tokenizer...")

    if args.model_path:
        # Construct full path by joining model_path and model name
        full_model_path = os.path.join(args.model_path, args.model)
        logger.info(f"Loading local model from {full_model_path}...")

        # Validate path exists
        if not os.path.exists(full_model_path):
            raise ValueError(f"Model not found at: {full_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = LlamaForCausalLM.from_pretrained(full_model_path, device_map="auto")
    else:
        # Load directly from HuggingFace
        logger.info(f"Loading model {args.model} from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto")

    return tokenizer, model


def main(args):
    tokenizer, model = load_model(args)

    # Get topics either from list or directory
    if args.topic_list:
        topics = read_topic_list(args.topic_list)
    else:
        topics = get_topics_from_directory(args.input_dir)
    
    logger.info(f"Found {len(topics)} topics to evaluate")

    all_results = {}
    rubric_totals = {}  # New dict to track totals for each rubric
    rubric_counts = {}  # New dict to track counts for each rubric

    for topic in topics:
        logger.info(f"Processing topic: {topic}")

        # Construct path for this topic's input file
        topic_file = os.path.join(args.input_dir, f"{topic}.txt")
        if not os.path.exists(topic_file):
            logger.warning(f"Skipping topic '{topic}': File not found at {topic_file}")
            continue

        # Read and preprocess the response for this topic
        response = preprocess_text(process_document(topic_file, max_words=10000))

        # Get grading for single response
        grading = get_grading_dict(
            responses=[response],
            topic=topic,
            tokenizer=tokenizer,
            model=model,
            prompt_template_path=args.prompt_template_path,
            rubric_path=args.rubric_path,
            disable_sample=args.disable_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            logger=logger,
        )

        # Store results for this topic
        all_results[topic] = grading

        # Accumulate scores for each rubric
        for rubric, results in grading.items():
            if rubric not in rubric_totals:
                rubric_totals[rubric] = 0
                rubric_counts[rubric] = 0

            # Get the score from the first (and only) response
            try:
                score = int(results[0]["score"])
                rubric_totals[rubric] += score
                rubric_counts[rubric] += 1
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(
                    f"Could not process score for topic {topic}, rubric {rubric}: {e}"
                )

    # Calculate averages for each rubric
    rubric_averages = {
        rubric: rubric_totals[rubric] / rubric_counts[rubric]
        for rubric in rubric_totals
        if rubric_counts[rubric] > 0
    }

    # Prepare output paths
    output_dir = os.path.dirname(args.output_path)
    output_base = os.path.splitext(args.output_path)[0]
    detailed_output_path = f"{output_base}_detailed.json"
    average_output_path = f"{output_base}_averages.json"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    with open(detailed_output_path, "w") as outfile:
        json.dump(all_results, outfile, indent=2)
        logger.info("Detailed results saved to: %s", detailed_output_path)

    # Save average scores
    with open(average_output_path, "w") as outfile:
        json.dump({"rubric_averages": rubric_averages}, outfile, indent=2)
        logger.info("Average scores saved to: %s", average_output_path)


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser(description="Evaluate text files based on topics.")
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Directory containing topic text files (topic.txt)",
    )
    parser.add_argument(
        "-l",
        "--topic_list",
        required=False,
        help="Optional: Path to file containing list of topics (one per line). If not provided, will process all .txt files in input_dir",
    )
    parser.add_argument(
        "-o", "--output_path", required=True, help="Path to save the output JSON file"
    )

    parser.add_argument(
        "--prompt_template_path",
        default="./eval_prometheus_no_ref.prompt",
        help="path to evaluation prometheus prompt template",
    )
    parser.add_argument(
        "--rubric_path",
        default="./eval_rubric_5.json",
        help="path to rubric json file",
    )

    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument(
        "--model",
        choices=["kaist-ai/prometheus-13b-v1.0", "kaist-ai/prometheus-7b-v1.0", "prometheus-eval/prometheus-7b-v2.0"],
        default="kaist-ai/prometheus-13b-v1.0",
        help="Model name to use. If model_path provided, will look for model in that directory",
    )
    parser.add_argument(
        "--disable_sample",
        action="store_true",
        help="Whether to disable sampling; default is False",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature for generation; default is 0.01",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top P for generation; default is 0.95",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate; default is 512",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.03,
        help="Repetition penalty; default is 1.03",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Base path to model directory. Will be combined with --model",
    )

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    # Update validation
    if args.model_path:
        assert os.path.exists(
            args.model_path
        ), f"model_path: {args.model_path} does not exist"
        assert os.path.isdir(
            args.model_path
        ), f"model_path: {args.model_path} is not a directory"

    # Validation checks
    assert os.path.exists(args.input_dir), f"input_dir: {args.input_dir} does not exist"
    if args.topic_list:
        assert os.path.exists(
            args.topic_list
        ), f"topic_list: {args.topic_list} does not exist"

    # Check if the file exists and ask for user confirmation to override
    if os.path.exists(args.output_path):
        overwrite = input(
            f"The file {args.output_path} already exists. Do you want to overwrite it? (y/n): "
        )
        if overwrite.lower() != "y":
            logger.info("User chose not to overwrite the existing file. Exiting.")
            exit()

    main(args)
