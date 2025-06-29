import dspy
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append("..")
sys.path.append("./src")

from lm import OpenAILM
from config import Config

# class Rewriter(dspy.Signature):
#     """Given an article, rewrite it to convert the structured text (List or Tabular data) into natural paragraphs to improve coherence.
#     Guidelines:
#     1. Change as few words as possible
#     2. For sections that do not have any content, remove them
#     3. Keep the original markdown format
#     4. For structured data, convert it into natural paragraphs. For example, if you have a bullet point that says "- <Song name>", convert it into a sentence like "The songs by... include <Song name>."
#     """
#     original_article = dspy.InputField(prefix="Original Article")
#     rewritten_article = dspy.OutputField(prefix="Rewritten Article")

# class Rewriter(dspy.Signature):
#     """Rewrite the given article to improve coherence and readability.
#     Guidelines:
#     1. Remove sections that do not have any content or has a very short (one to two sentences) content.
#     2. Remove every structured data (list, table, etc.) and their corresponding section to improve readability.
#     3. Keep the original markdown format.
#     4. Change as few words as possible.
#     """
#     original_article = dspy.InputField(prefix="Original Article")
#     rewritten_article = dspy.OutputField(prefix="Rewritten Article")

class Rewriter(dspy.Signature):
    """
    Refine the given article to improve coherence and readability.
    Guidelines:
    1. Keep the markdown format.
    2. Retain as much information as possible.
    """
    original_article = dspy.InputField(prefix="Original Article")
    rewritten_article = dspy.OutputField(prefix="Rewritten Article")


def setup_language_model(model_name: str, max_tokens: int):
    """Create and return the language model we'll use."""
    common_params = {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_BASE_URL"],
        "max_tokens": max_tokens,
        "support_structured_response": True,
    }
    return OpenAILM(model=model_name, **common_params)


def main(input_dir: str, output_dir: str):
    """Rewrite articles in the input directory and save to output directory."""
    # Initialize configuration and models
    config = Config()
    lm = setup_language_model("gpt-4o-2024-08-06", 16000)
    dspy.configure(lm=lm)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize rewriter
    rewrite_function = dspy.Predict(Rewriter, max_tokens=16000)

    # Process all text files
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for file in tqdm(files, desc="Rewriting articles"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        try:
            with open(input_path, "r") as f:
                text = f.read()
                rewritten_text = rewrite_function(original_article=text).rewritten_article
                
            with open(output_path, "w") as f:
                f.write(rewritten_text)
                
            print(f"Successfully rewritten: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrite articles to improve coherence and readability")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing text files to rewrite"
    )
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Output directory for rewritten articles"
    )
    
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)


