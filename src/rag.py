import argparse
import dspy
import uuid
from typing import Optional, Union
import json
from pydantic import BaseModel
import re
import os
import os.path

from article import Article, Outline
from agents.retriever import ValueSERPHandler
from memory import Memory, MemoryUnit
from utils import Logger, Parser
from config import Config
from lm import OpenAILM

logger = Logger("baseline")


class OutlineStructure(BaseModel):
    section_list: list[str]


class Outliner(dspy.Signature):
    """You are an experienced Wikipedian tasked with creating a general, one-level outline for a Wikipedia article section
    Guidelines:
    1. Provide only main subsection headings, without any further subdivisions.
    2. Ensure comprehensive coverage of key aspects related to the section title.
    3. Focus on the most important aspects within the section.
    4. Follow Wikipedia's style and naming conventions."""

    section_title = dspy.InputField(prefix="Section title: ", format=str)
    outline = dspy.OutputField(prefix="List of the subsection headings: ", format=str)


class OutlineRewriter(dspy.Signature):
    """You are an experienced Wikipedian tasked with improving an existing one-level outline for a Wikipedia article section.
    Guidelines:
    1. Provide only main subsection headings, without any further subdivisions.
    2. Maintain the original structure of the original outline.
    3. Add general headings, and delete unreasonable headings according to the collected information.
    4. Focus on the most important aspects within the section.
    5. Follow Wikipedia's style and naming conventions.
    """

    section_title = dspy.InputField(prefix="Section title: ", format=str)
    information_collected = dspy.InputField(
        prefix="The information collected", format=str
    )
    current_outline = dspy.InputField(prefix="Current outline:\n", format=list[str])
    outline = dspy.OutputField(prefix="List of the subsection headings: ", format=str)


class WriteOutline(dspy.Module):
    """Generate the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_outline = dspy.Predict(
            Outliner, **{"response_format": OutlineStructure}
        )
        self.rewrite_outline = dspy.Predict(
            OutlineRewriter, **{"response_format": OutlineStructure}
        )
        self.engine = engine

    def forward(
        self,
        topic: str,
        current_outline: Optional[Outline] = None,
        memo: Optional[Union[str, list[str]]] = None,
    ):
        with dspy.settings.context(lm=self.engine):
            if current_outline is None:  # first run
                outline_result = self.draft_outline(section_title=topic).outline
                outline_list = json.loads(str(outline_result))["section_list"]
            else:
                outline_list = current_outline.to_flatten_list()

            outline = self.rewrite_outline(
                section_title=topic,
                information_collected=memo,
                current_outline=outline_list,
            ).outline
            outline_list = json.loads(str(outline))["section_list"]

        return dspy.Prediction(outline_list=outline_list)


# Adapted from STORM-NAACL 2024
class SectionWriter(dspy.Signature):
    """Write a Wikipedia section based on the collected information.

    Here is the format of your writing:
        1. Use "#" Title" to indicate section title, "##" Title" to indicate subsection title, "###" Title" to indicate subsubsection title, and so on.
        2. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
    """

    info = dspy.InputField(prefix="The collected information:\n", format=str)
    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    section = dspy.InputField(prefix="The section you need to write: ", format=str)
    output = dspy.OutputField(
        prefix="Write the section with proper inline citations (Start your writing with # section title. Don't include the page tile or try to write other sections):\n"
    )


class WriteSection(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(SectionWriter)
        self.engine = engine

    def forward(
        self, topic: str, section_name: str, memory_unit_list: list[MemoryUnit]
    ):
        info = ""
        for i, memory_unit in enumerate(memory_unit_list):
            info += f"[{i+1}] {memory_unit.content}\n"
        with dspy.settings.context(lm=self.engine):
            return self.write_section(
                info=info, topic=topic, section=section_name
            ).output


class RAG:
    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        topic: str,
        config: Config,
    ):
        self.engine = engine
        self.topic = topic
        self.config = config
        self.memory = Memory(
            topic=topic, config=config, force_recreate=True
        )
        self.url_set = set()

        self.write_outline = WriteOutline(engine=self.engine)
        self.write_section = WriteSection(engine=self.engine)

    def retrieve_related_webpage(self, query: str):
        search_handler = ValueSERPHandler(self.config)
        webpage_dict = search_handler.search(query)

        logger.info(f"URLs found: {webpage_dict.keys()}")

        memory_unit_list = []
        for url, webpage in webpage_dict.items():
            if url in self.url_set:
                continue
            self.url_set.add(url)
            webpage_text = webpage["text"]
            webpage_chunk_list = Parser.split_doc_into_chunks(webpage_text)
            for chunk in webpage_chunk_list:
                memory_unit = MemoryUnit(uuid=str(uuid.uuid4()), content=chunk, url=url)
                memory_unit_list.append(memory_unit)

        self.memory.add_information_in_batch(
            memory_unit_list=memory_unit_list,
            custom_metadata={"query": query},
            skip_similar=False,
        )

        return memory_unit_list

    def plan_outline(self, topic: str):
        outline = Outline(title=topic)
        outline_list = self.write_outline(
            topic=topic,
            memo=self.memory.retrieve_information(query=topic),
            current_outline=None,
        ).outline_list

        parsed_outline_list = [Parser.parse_section_name(name) for name in outline_list]
        for section_name in parsed_outline_list:
            if section_name.lower() not in self.config.EXCLUDE_SECTION_LIST:
                outline.create_child(section_name)

        return outline

    def generate_section(self, section_name: str, root_article: Article):
        relevant_memory_unit_list = self.memory.retrieve_information(query=section_name)
        section_text = self.write_section(
            topic=self.topic,
            section_name=section_name,
            memory_unit_list=relevant_memory_unit_list,
        )
        section = Article.from_text(
            title=section_name.split("//")[-1],
            text=section_text,
            layer=1,
            working_context=relevant_memory_unit_list,
            keep_citation_numbers=True,
        )
        for paragraph in section.paragraph_list:
            for sentence in paragraph.sentence_list:
                cite_id_list = re.findall(r"\[(\d+)\]", sentence.content)
                cite_id_list = [int(cite_id) for cite_id in cite_id_list]
                sentence.citation_list = []
                sentence.doc_id_list = []
                for cite_id in cite_id_list:
                    try:
                        memory_unit = relevant_memory_unit_list[cite_id - 1]
                        sentence.citation_list.append(memory_unit)
                        doc_id = root_article.record_cited_mu(memory_unit)
                        sentence.doc_id_list.append(doc_id)
                    except IndexError:
                        logger.warning(
                            f"Citation {cite_id} not found in {section_name}, {len(relevant_memory_unit_list)}"
                        )

        root_article.add_subsection(section)
        return section

    def execute(self):
        memory_unit_list = self.retrieve_related_webpage(self.topic)
        outline = self.plan_outline(topic=self.topic)
        logger.info(f"Outline: {outline}")
        for section_name in outline.to_flatten_list():
            new_query = f"{self.topic} {section_name.split('//')[-1]}"
            memory_unit_list = self.retrieve_related_webpage(new_query)

        print(f"Total information: {self.memory.count_information()}")

        article = Article(title=self.topic)
        for section_name in outline.to_flatten_list():
            self.generate_section(section_name=section_name, root_article=article)

        rag_dir = self.config.OUTPUT_DIR
        os.makedirs(rag_dir, exist_ok=True)
        article.save_file(rag_dir)
        logger.info(f"Article {self.topic} finished")
        return article


def main(args):
    config = Config()

    with open(args.file_path, "r") as f:
        topics = f.readlines()

    lm = OpenAILM(
        model=config.MODEL_NAME,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_tokens=config.MAX_TOKENS,
        support_structured_response=True,
    )

    better_lm = OpenAILM(
        model="gpt-4o-2024-08-06",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_tokens=4000,
        support_structured_response=True,
    )

    for topic in topics:
        topic = topic.strip()
        output_path = os.path.join(
            config.OUTPUT_DIR, "raw_txt", f"{Parser.safe_title(topic)}.txt"
        )

        if os.path.exists(output_path):
            logger.info(f"Skipping {topic} - output already exists")
            continue

        rag = RAG(engine=better_lm, topic=topic, config=config)
        article = rag.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the planner with optional skip flags."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="data/topic_list.txt",
        help="The file containing the topics",
    )
    args = parser.parse_args()
    main(args)
