import concurrent.futures
import dspy
from typing import Union, Optional
import json
from memory import Memory
from article import Outline, Article
from utils import Logger, Config
from pydantic import BaseModel

logger = Logger(__name__)


class WriterAgent:
    def __init__(
        self,
        topic: str,
        outline: Outline,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        better_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        memory: Memory,
        config: Config,
    ):
        # Store basic parameters
        self.topic = topic
        self.outline = outline
        self.memory = memory
        self.config = config
        self.MIN_MEMORY_UNIT_FOR_WRITING = config.MIN_MEMORY_UNIT_FOR_WRITING

        # Initialize LM-based components
        self.generate_section_content = WriteSection(engine=better_engine)
        self.assign_citation = CiteMemoryUnit(engine=engine)

    def add_citation(self, article: Article):
        """Add citations to an article by matching sentences to memory units."""

        def process_sentence(sentence, working_context):
            """Match a single sentence to relevant memory units and add citations."""
            # Get relevant memory unit indices for the sentence
            citation_indices = self.assign_citation(
                sentence=sentence.content,
                memory_unit_list=[mu.content for mu in working_context],
            )

            # Add citations to the sentence
            sentence.citation_list = []
            sentence.doc_id_list = []
            for idx in citation_indices:
                try:
                    memory_unit = working_context[idx]
                    sentence.citation_list.append(memory_unit)
                    doc_id = article.record_cited_mu(memory_unit)
                    sentence.doc_id_list.append(doc_id)
                except:
                    logger.error(
                        f"Failed to add citation {idx} for sentence: {sentence.content}"
                    )

        # Process all sentences concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for paragraph in article.paragraph_list:
                for sentence in paragraph.sentence_list:
                    futures.append(
                        executor.submit(
                            process_sentence, sentence, article.working_context
                        )
                    )
            concurrent.futures.wait(futures)

        # Recursively process subsections
        for section in article.subsection_list:
            self.add_citation(section)

        return article

    def write_section(self, full_section_name: str, layer: int) -> Optional[Article]:
        """Write a single section using retrieved memory units."""
        section_name = full_section_name.split("//")[-1]
        memory_unit_list = self.memory.retrieve_information(
            section_name,
            constraint={"tag": full_section_name},
            k=100,
        )

        if len(memory_unit_list) > self.MIN_MEMORY_UNIT_FOR_WRITING:
            # Generate section content from memory units
            section_content = self.generate_section_content(
                topic=self.topic,
                section_name=full_section_name,
                fact_list=memory_unit_list,
            ).content

            return Article.from_text(
                title=section_name,
                text=section_content,
                layer=layer,
                working_context=memory_unit_list,
            )
        return None

    def write_all_section(
        self,
        outline: Outline,
        section_title: str,
    ) -> Article:
        """Write all sections recursively following the outline structure."""
        # Initialize root article
        root_article = Article(title=outline.title, layer=outline.layer)

        # Process subsections concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_child = {}
            for sub_outline in outline.children:
                full_section_title = f"{section_title}//{sub_outline.title}"

                # Submit appropriate task based on outline structure
                if not sub_outline.children:
                    # For leaf sections, write the section content
                    future = executor.submit(
                        self.write_section, full_section_title, sub_outline.layer
                    )
                else:
                    # For non-leaf sections, recursively process subsections
                    future = executor.submit(
                        self.write_all_section, sub_outline, full_section_title
                    )
                future_to_child[future] = sub_outline

            concurrent.futures.wait(future_to_child)

        # Add subsections in original order
        for sub_outline in outline.children:
            for future, child_outline in future_to_child.items():
                if child_outline != sub_outline:
                    continue

                section = future.result()
                if section is not None:
                    logger.info(f"Section {sub_outline.title}:\n{section}")
                    root_article.add_subsection(section)
                else:
                    logger.error(f"Section {sub_outline.title} is empty.")
                break

        return root_article

    def write_lead_section(self) -> Article:
        """Write the overview/lead section of the article."""
        memory_unit_list = self.memory.retrieve_information(
            f"Overview of {self.topic}", k=100
        )

        section_content = self.generate_section_content(
            topic=self.topic,
            section_name=f"Overview of {self.topic}",
            fact_list=memory_unit_list,
        ).content

        return Article.from_text(
            title=self.topic,
            text=section_content,
            layer=0,
            working_context=memory_unit_list,
        )

    def execute(self) -> Article:
        """Execute the full article writing process."""

        article = self.write_all_section(self.outline, self.topic)

        lead_section = self.write_lead_section()
        article.paragraph_list = lead_section.paragraph_list
        article.working_context = lead_section.working_context

        cited_article = self.add_citation(article=article)
        cited_article.save_file(self.config.OUTPUT_DIR)

        return cited_article


class SectionWriter(dspy.Signature):
    """Write a Wikipedia section based on atomic facts. Do not improvise with any other information.
    1. Do not include section name as output.
    2. Ensure the section is split into multiple coherent paragraphs if necessary.
    3. The sequence of facts must be adjusted for coherence and readability.
    4. Stay focused within the section title, only include facts related to the section title.
    """

    topic = dspy.InputField(prefix="Topic: ", format=str)
    section_name = dspy.InputField(prefix="Section title: ", format=str)
    fact_list = dspy.InputField(prefix="Atomic facts: ", format=list[str])
    content = dspy.OutputField(prefix="Section content: ", format=str)


class CitationIndexList(BaseModel):
    index_list: list[int]


class CitationWriter(dspy.Signature):
    """Given a sentence and a list of memory units, only cite the most relevant memory units to fully cover the sentence.
    The output should be a list of indices of the memory units in the input list.
    """

    sentence = dspy.InputField(prefix="Sentence: ")
    memory_unit_list = dspy.InputField(prefix="Memory units: ", format=list[str])
    index_list = dspy.OutputField(prefix="Indices of assigned memory units: ")


class CiteMemoryUnit(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.engine = engine
        self.assign_memory_unit = dspy.Predict(
            CitationWriter, **{"response_format": CitationIndexList}
        )

    def forward(self, sentence: str, memory_unit_list: list[str]):
        with dspy.settings.context(lm=self.engine):
            parsed_memory_unit_list = [
                f"{idx+1}. {mu}" for idx, mu in enumerate(memory_unit_list)
            ]
            index_list = self.assign_memory_unit(
                sentence=sentence, memory_unit_list=parsed_memory_unit_list
            ).index_list
            index_list = json.loads(index_list)["index_list"]
            try:
                index_list = [idx - 1 for idx in index_list]
            except:
                logger.error(f"Failed to parse index list: {index_list}")
                index_list = []

            return index_list


class WriteSection(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.write_section = dspy.Predict(SectionWriter)
        self.engine = engine

    def forward(self, topic: str, section_name: str, fact_list: list[str]):
        with dspy.settings.context(lm=self.engine):
            content = self.write_section(
                topic=topic,
                section_name=section_name,
                fact_list=fact_list,
            ).content

            if "Section content:" in content:
                content = content.split("Section content:", 1)[1].strip()

        return dspy.Prediction(content=content)
