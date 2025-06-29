import concurrent.futures
import dspy
from typing import Union, Optional
from memory import Memory
from article import Outline, Article
from utils import Logger
from config import Config

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

        # Store language models for citation
        self.lm = engine
        self.better_lm = better_engine

        # Initialize LM-based components
        self.generate_section_content = WriteSection(engine=better_engine)
        self.citation_finder = dspy.Predict(CitationFinder)

    def add_citation(self, article: Article):
        """Add citations to an article by matching sentences to memory units."""
        
        def process_sentence(sentence, working_context, root_article):
            """Process a single sentence to find and add citations."""
            if not sentence.citation_list:
                sentence.citation_list = []
                # Create text with numbered sources
                mu_text = ""
                for idx, memory_unit in enumerate(working_context):
                    mu_text += f"{idx+1}: {memory_unit.content}\n"

                # Find relevant source indices
                with dspy.context(lm=self.lm):
                    try:
                        answer = self.citation_finder(
                            claim=sentence.content, source_list=mu_text
                        )
                        indices = [int(idx) for idx in answer.answer.split(", ")]
                        for idx in indices:
                            if 0 <= idx - 1 < len(working_context):
                                sentence.citation_list.append(
                                    working_context[idx - 1]
                                )
                    except Exception as e:
                        try:
                            with dspy.context(lm=self.better_lm):
                                answer = self.citation_finder(
                                    claim=sentence.content, source_list=mu_text
                                )
                                indices = [
                                    int(idx) for idx in answer.answer.split(", ")
                                ]
                                for idx in indices:
                                    if 0 <= idx - 1 < len(working_context):
                                        sentence.citation_list.append(
                                            working_context[idx - 1]
                                        )
                        except Exception as e:
                            logger.error(
                                f"Error processing sentence: {sentence.content}\nError: {e}"
                            )
                            return False  # Mark as failed

            sentence.doc_id_list = []
            for mu in sentence.citation_list:
                doc_id = root_article.record_cited_mu(mu)
                sentence.doc_id_list.append(doc_id)
            return True

        total_sentences = 0
        uncited_sentences = 0
        
        # Collect all sentences from all sections (including subsections)
        def collect_sentences(section):
            sentences = []
            for paragraph in section.paragraph_list:
                for sentence in paragraph.sentence_list:
                    sentences.append((sentence, section.working_context))
            
            for subsection in section.subsection_list:
                sentences.extend(collect_sentences(subsection))
            
            return sentences

        all_sentences = collect_sentences(article)
        total_sentences = len(all_sentences)

        # Process all sentences concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_sentence, sentence, working_context, article)
                for sentence, working_context in all_sentences
            ]
            
            # Wait for all futures to complete and count failures
            for future, (sentence, _) in zip(futures, all_sentences):
                try:
                    success = future.result()
                    if not success:
                        uncited_sentences += 1
                except Exception as e:
                    logger.error(f"Failed to process sentence: {sentence.content}\nError: {e}")
                    uncited_sentences += 1

        return total_sentences, uncited_sentences

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

        # Add citations and get statistics
        total_sentences, uncited_sentences = self.add_citation(article=article)
        citation_ratio = (
            1 - (uncited_sentences / total_sentences) if total_sentences > 0 else 0
        )
        logger.info(
            f"Citation ratio: {citation_ratio:.2%} ({total_sentences - uncited_sentences}/{total_sentences} sentences cited)"
        )

        article.save_file(self.config.OUTPUT_DIR)

        return article

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
