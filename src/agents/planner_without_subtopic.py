import dspy
from typing import Union, Optional
import concurrent.futures
from pydantic import BaseModel
import json
import os
import re

from agents.retriever import RetrieverAgent, Summarize
from agents.writer import WriterAgent
from memory import Memory
from utils import Logger, Parser, Config
from article import Outline

logger = Logger("planner")


class Planner:
    def __init__(
        self,
        topic: str,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        better_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        config: Config,
    ):
        """Initialize the Planner agent."""
        self.topic = topic
        self.engine = engine
        self.better_engine = better_engine
        self.config = config
        self.memory = Memory(
            topic=self.topic,
            config=self.config,
            engine=self.engine,
            name="mog_wo_subtopic",
        )

        # Initialize sub-agents
        self.summarize = Summarize(engine=self.engine)
        self.write_outline = WriteOutline(engine=self.better_engine)
        self.refine_outline = RefineOutline(engine=self.better_engine)

    def bootstrap(self) -> str:
        """Bootstrap the planning process by getting initial facts and summary."""
        # Get initial facts about the topic
        ra = RetrieverAgent(
            topic=self.topic, engine=self.engine, memory=self.memory, config=self.config
        )
        fact_list = ra.retrieve(query=self.topic)

        # Generate summary from facts
        summary = self.summarize(topic=self.topic, fact_list=fact_list).summary
        logger.info(f"Summary: {summary}")
        return summary

    def plan_outline(
        self,
        topic: str,
        current_outline: Optional[Outline] = None,
        memo: Optional[Union[str, list[dict]]] = None,
    ) -> Outline:
        """Plan the article outline based on topic and optional existing outline."""
        outline = Outline(title=topic)
        outline_list = self.write_outline(
            topic=topic,
            memo=memo,
            current_outline=current_outline,
        ).outline_list

        parsed_outline_list = [Parser.parse_section_name(name) for name in outline_list]
        for section_name in parsed_outline_list:
            if section_name.lower() not in self.config.EXCLUDE_SECTION_LIST:
                outline.create_child(section_name)

        return outline

    def collect_information(self, outline: Outline) -> list:
        """Collect information for each section in parallel."""

        def _collect_information(section_name: str) -> list:
            logger.info(f"Collecting information for section: {section_name}")
            ra = RetrieverAgent(
                topic=section_name,
                engine=self.engine,
                memory=self.memory,
                config=self.config,
            )
            section_memory_list = ra.retrieve(
                query=f"{section_name.replace('//', ' ')}"
            )
            logger.info(
                f"{len(section_memory_list)} memory units collected for section {section_name}"
            )
            return section_memory_list

        # Collect information for all sections in parallel
        memory_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_section = {
                executor.submit(_collect_information, section_name): section_name
                for section_name in outline.to_flatten_list()
            }

            for future in concurrent.futures.as_completed(future_to_section):
                section_name = future_to_section[future]
                memory_list.extend(future.result())

        return memory_list

    def recursive_outline_planning(
        self, full_topic: str, depth: int = 0
    ) -> Optional[Outline]:
        """Recursively plan the outline, grouping related information."""
        outline = Outline(title=full_topic.split("//")[-1])

        # Set constraints based on depth
        constraint = {"tag": full_topic} if depth > 0 else {}

        # Check if we should continue exploring
        memory_unit_cnt = self.memory.count_information(constraint=constraint)
        if (
            memory_unit_cnt < self.config.MIN_MEMORY_UNIT_TO_EXPLORE
            or depth >= self.config.MAX_WRITE_DEPTH
        ):
            return outline

        # Group and summarize information clusters
        clusters = self.memory.group_information(constraint=constraint)
        cluster_summary_list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_cluster = {
                executor.submit(
                    self.summarize,
                    topic=full_topic,
                    fact_list=information_cluster[
                        : self.config.MAX_CLUSTER_SUMMARIZE_MEMORY_UNIT
                    ],
                ): cluster_id
                for cluster_id, information_cluster in clusters.items()
            }

            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                cluster_summary = future.result().summary
                cluster_summary_list.append(cluster_summary)

        # Plan and refine outline structure
        new_outline = self.plan_outline(topic=full_topic, memo=cluster_summary_list)
        subsection_list = new_outline.to_flatten_list()
        subsections_to_keep = self.memory.label_information(
            parent_path=full_topic,
            label_list=[l.split("//")[-1] for l in subsection_list],
            constraint=constraint,
        )

        # Recursively process subsections
        for subsection_title in subsections_to_keep:
            full_subsection_title = f"{full_topic}//{subsection_title}"
            subsection_outline = self.recursive_outline_planning(
                full_topic=full_subsection_title, depth=depth + 1
            )
            if subsection_outline:
                outline.insert_child(subsection_outline)
            else:
                # Move information to parent section if subsection is discarded
                self.memory.update_information(
                    constraint={"tag": full_subsection_title},
                    new_metadata={"tag": full_topic},
                )

        return outline

    def execute(self, skip_research: bool = False, skip_outline: bool = False):
        """Execute the full planning and writing process."""
        # Research phase
        if not skip_research:
            self.summary = self.bootstrap()
            logger.info(f"Summary: {self.summary}")

            self.outline = self.plan_outline(topic=self.topic, memo=self.summary)
            logger.info(f"Outline: {self.outline}")

            memory_list = self.collect_information(self.outline)
        else:
            logger.info("Skip research")

        logger.info(f"{self.memory.collection.count()} memory units in total")

        # Outline planning phase
        dir = os.path.join(self.config.OUTPUT_DIR, "outline")
        os.makedirs(dir, exist_ok=True)

        if not skip_outline:
            self.outline_from_memory = self.recursive_outline_planning(
                full_topic=self.topic
            )
            logger.info(f"Outline from memory: {self.outline_from_memory}")

            refined_outline = self.refine_outline(
                outline=self.outline_from_memory.to_dict()
            ).outline
            self.refined_outline = Outline.from_dict(refined_outline)

            with open(f"{dir}/{Parser.safe_title(self.topic)}.json", "w") as f:
                json.dump(self.refined_outline.to_dict(), f)
            logger.info(f"Refined outline: {self.refined_outline}")
        else:
            self.refined_outline = Outline.from_dict(
                json.load(
                    open(
                        f"{dir}/{Parser.safe_title(self.topic)}.json",
                        "r",
                    )
                )
            )
            logger.info(f"Loaded outline: {self.refined_outline}")

        # Final outline refinement and writing
        flat_outline = self.memory.label_information(
            parent_path=None, label_list=self.refined_outline.to_flatten_list()
        )
        self.refined_outline = Outline.from_flatten_list(flat_outline)
        logger.info(f"Final outline after relabeling: {self.refined_outline}")

        wa = WriterAgent(
            topic=self.topic,
            outline=self.refined_outline,
            engine=self.engine,
            better_engine=self.better_engine,
            memory=self.memory,
            config=self.config,
        )

        article = wa.execute()
        logger.info(f"Article drafted: {article}")
        return article


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
        memo: Optional[Union[str, list[dict]]] = None,
    ):
        with dspy.settings.context(lm=self.engine):
            if current_outline is None:  # first run
                current_outline = self.draft_outline(section_title=topic).outline
                current_outline = json.loads(current_outline)["section_list"]
            else:
                current_outline = current_outline.to_flatten_list()

            outline = self.rewrite_outline(
                section_title=topic,
                information_collected=memo,
                current_outline=current_outline,
            ).outline
            outline = json.loads(outline)["section_list"]

        return dspy.Prediction(outline_list=outline)


class OutlineRefiner(dspy.Signature):
    """Refine the outline for the Wikipedia page.
    1. Remove redundant subsections.
    2. Keep all the sections.
    3. Do not add any new section."""

    outline = dspy.InputField(prefix="Outline: ")
    refined_outline = dspy.OutputField(prefix="Refined outline: ")


class RefineOutline(dspy.Module):
    """Refine the outline for the Wikipedia page."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.engine = engine
        self.refine_outline = dspy.Predict(
            OutlineRefiner,
            **{
                "response_format_default": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "outline",
                        "strict": True,
                        "description": "Hierarchical outline for the Wikipedia page",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "section_title": {
                                    "type": "string",
                                    "description": "The title of the section",
                                },
                                "subsection_list": {
                                    "type": "array",
                                    "description": "The list of subsections",
                                    "items": {"$ref": "#"},
                                },
                            },
                            "required": ["section_title", "subsection_list"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )

    def forward(self, outline: dict):
        with dspy.settings.context(lm=self.engine):
            refined_outline = self.refine_outline(outline=str(outline)).refined_outline
            refined_outline = json.loads(refined_outline)
        return dspy.Prediction(outline=refined_outline)
