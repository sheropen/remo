import dspy
import os
import sys
import toml
from tqdm import tqdm

sys.path.append("./src")

class Config:
    def __init__(self, config_path="config.toml", secret_path="secret.toml"):
        self.load_config(config_path=config_path)
        self.load_secret(secret_path=secret_path)

    def load_secret(self, secret_path):
        with open(secret_path, "r") as file:
            data = toml.load(file)
        for key, value in data.items():
            os.environ[key] = str(value)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config_data = toml.load(f)

        for key, value in config_data.items():
            if key.endswith("_DIR"):
                os.makedirs(value, exist_ok=True)
            setattr(self, key, value)

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

config = Config("config.toml")

lm = dspy.LM(model="gpt-4o-2024-08-06", api_key=os.environ["OPENAI_API_KEY"], api_base=os.environ["OPENAI_BASE_URL"])
dspy.configure(lm=lm)

folder = "/home/junhao/projects/mog/data/evaluation_dataset/fresh_wiki/clean_txt"
output_dir = "/home/junhao/projects/mog/data/evaluation_dataset/fresh_wiki/refine_clean_txt"
os.makedirs(output_dir, exist_ok=True)

rewrite_function = dspy.Predict(Rewriter, max_tokens=16000)

files = [f for f in os.listdir(folder) if f.endswith(".txt")]
for file in tqdm(files, desc="Rewriting articles"):
    with open(os.path.join(folder, file), "r") as f:
        text = f.read()
        rewritten_text = rewrite_function(original_article=text).rewritten_article
        with open(os.path.join(output_dir, file), "w") as f:
            f.write(rewritten_text)


