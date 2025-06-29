import concurrent.futures
import dspy
import json
import os
import requests
from typing import Union, List
import uuid
from pydantic import BaseModel


from memory import Memory, MemoryUnit
from utils import WebsiteContentProcessor, Parser, Logger
from config import Config

logger = Logger("retriever")


class ValueSERPHandler:
    """
    Handles search queries using the ValueSERP API and web page downloads.
    """

    def __init__(self, config: Config):
        self.config = config
        self.value_serp_api = os.environ.get("VALUESERP_API_KEY")
        self.webpage_helper = WebsiteContentProcessor(config=self.config)

    def serp(self, query: str) -> list:
        """
        Performs a search using ValueSERP API and returns results.
        """
        # Check cache first
        cache_path = os.path.join(
            self.config.SERP_DIR, f"{Parser.safe_title(query)}.json"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        # Make API request if not cached
        try:
            params = {
                "api_key": self.value_serp_api,
                "q": query,
                "num": 20,
            }
            response = requests.get(
                "https://api.valueserp.com/search", params=params
            ).json()

            # Filter and format results
            serp_results = [
                {
                    "title": result.get("title"),
                    "url": result.get("link"),
                    "snippet": result.get("snippet"),
                    "source": "google",
                }
                for result in response.get("organic_results", [])
                if not any(
                    exclude in result["link"]
                    for exclude in self.config.EXCLUDE_DOMAIN_LIST
                )
            ]

            # Cache results
            with open(cache_path, "w") as f:
                json.dump(serp_results, f)

            return serp_results

        except Exception as e:
            logger.error(f"Error in serp: {e}")
            return []

    def search(self, query: str) -> dict:
        """
        Performs search and extracts content from result pages.
        """
        serp_results = self.serp(query)
        urls = [result["url"] for result in serp_results]
        return self.webpage_helper.extract_content_chunks(urls)


class RetrieverAgent:
    def __init__(
        self,
        topic,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        memory: Memory,
        config: Config,
    ):
        self.config = config

        self.topic = topic
        self.search_handler = ValueSERPHandler(config=self.config)
        self.memory = memory

        self.extract_memory_unit = ExtractFactList(engine=engine)
        self.explore_subtopic = ExploreSubtopic(engine=engine)
        self.convert_topic_to_search_query = ConvertTopicToSearchQuery(engine=engine)
        self.summarize_memory_units = Summarize(engine=engine)

    def retrieve(self, query: str) -> List[MemoryUnit]:
        """
        Retrieves and processes webpage content for a given search query.
        Returns a list of MemoryUnit objects containing extracted facts.
        """
        # Search for relevant webpages
        webpage_dict = self.search_handler.search(query)
        logger.info(f"URLs found: {webpage_dict.keys()}")

        def extract_facts_from_webpage(url: str, webpage: dict) -> List[MemoryUnit]:
            """Extract facts from a single webpage and convert to MemoryUnits"""
            facts = self.extract_memory_unit(
                topic=self.topic, text=webpage["text"]
            ).fact_list

            return [
                MemoryUnit(uuid=str(uuid.uuid4()), content=fact, url=url)
                for fact in facts
            ]

        # Process webpages concurrently
        all_memory_units = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit webpage processing tasks
            future_to_url = {
                executor.submit(extract_facts_from_webpage, url, webpage): url
                for url, webpage in list(webpage_dict.items())[
                    : self.config.MAX_WEBPAGE
                ]
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    webpage_memory_units = future.result()
                    all_memory_units.extend(webpage_memory_units)
                except Exception as exc:
                    logger.error(f"Failed to process {url}: {exc}")

        # Store extracted information
        self.memory.add_information_in_batch(
            memory_unit_list=all_memory_units, custom_metadata={"query": query}
        )

        return all_memory_units

    def research(self, topic: str, parent_summary: str, layer: int = 1) -> list[dict]:
        """
        Recursively research a topic and its subtopics, gathering facts and summaries.

        Args:
            topic: The topic to research
            parent_summary: Summary of the parent topic
            layer: Current depth in the recursive search (default=1)

        Returns:
            List of memo dictionaries containing topic summaries
        """
        memo_list = []

        # Generate and execute search queries
        search_queries = self.convert_topic_to_search_query(
            topic=topic, summary=parent_summary
        ).search_queries_list
        logger.info(f"Topic: {topic}, Search queries: {search_queries}")

        # Retrieve facts for each search query concurrently
        all_memory_units = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(self.retrieve, query): query
                for query in search_queries[: self.config.MAX_QUERY]
            }
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                memory_units = future.result()
                all_memory_units.extend(memory_units)

        if all_memory_units:
            # Summarize collected facts
            summary = self.summarize_memory_units(
                topic=topic, fact_list=all_memory_units
            ).summary
            memo_list.append({"topic": topic, "summary": summary})

            # Get subtopics for further exploration
            subtopics = self.explore_subtopic(
                topic=topic, summary=summary
            ).subtopic_list

            # Recursively research subtopics if not at max depth
            if layer < self.config.MAX_SEARCH_DEPTH:
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_subtopic = {
                        executor.submit(
                            self.research, f"{topic}//{subtopic}", summary, layer + 1
                        ): subtopic
                        for subtopic in subtopics[: self.config.MAX_SUBTOPIC]
                    }

                    for future in concurrent.futures.as_completed(future_to_subtopic):
                        subtopic = future_to_subtopic[future]
                        try:
                            subtopic_memos = future.result()
                            memo_list.extend(subtopic_memos)
                        except Exception as exc:
                            logger.error(
                                f"Subtopic {subtopic} generated an exception: {exc}"
                            )

        return memo_list


class AtomicFactExtractor(dspy.Signature):
    """
    Extract atomic facts strictly about a specific topic from the given text.

    Guidelines:
    1. Be explicit with name entities and time; avoid using pronouns such as "he", "she", "it", "they", or "the...".
    2. If the text is not about the topic, return an empty list "[]".
    """

    topic = dspy.InputField(desc="Specific topic to extract facts about")
    text = dspy.InputField(desc="Text that may contain information about the topic")
    fact_list = dspy.OutputField(desc="List of atomic facts strictly about the topic")


class FactStructure(BaseModel):
    fact_list: list[str]


class ExtractFactList(dspy.Module):
    """Extract a list of atomic facts about a specific topic from text."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.extract_fact = dspy.Predict(
            AtomicFactExtractor, **{"response_format": FactStructure}
        )
        self.engine = engine

    def forward(self, topic: str, text: str):
        with dspy.settings.context(lm=self.engine):
            combined_fact_list = []
            for chunk in Parser.split_doc_into_chunks(text, max_chunk_len=3000):
                extracted_fact_list = self.extract_fact(
                    text=chunk, topic=topic
                ).fact_list

                extracted_fact_list = json.loads(extracted_fact_list)["fact_list"]
                combined_fact_list += extracted_fact_list

        return dspy.Prediction(fact_list=combined_fact_list)


class SubtopicStructure(BaseModel):
    subtopic_list: list[str]


class SearchQueriesStructure(BaseModel):
    search_queries_list: list[str]


class SubtopicExplorer(dspy.Signature):
    """topic -> diverse subtopics"""

    topic = dspy.InputField(desc="Topic")
    summary = dspy.InputField(desc="Summary of the topic")
    subtopics = dspy.OutputField(
        desc="List of diverse subtopics related to the specific topic"
    )


class SearchQueriesExplorer(dspy.Signature):
    """topic -> diverse google search queries"""

    topic = dspy.InputField(desc="Topic")
    summary = dspy.InputField(desc="Summary of the topic")
    search_queries = dspy.OutputField(
        desc="List of diverse google search queries related to the specific topic"
    )


class ExploreSubtopic(dspy.Module):
    """Explore diverse research areas within a given topic."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.research_area = dspy.Predict(
            SubtopicExplorer, **{"response_format": SubtopicStructure}
        )
        self.engine = engine

    def forward(self, topic: str, summary: str):
        with dspy.settings.context(lm=self.engine):
            subtopic_list = self.research_area(topic=topic, summary=summary).subtopics
            subtopic_list = json.loads(subtopic_list)["subtopic_list"]

        return dspy.Prediction(subtopic_list=subtopic_list)


class ConvertTopicToSearchQuery(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.search_queries = dspy.Predict(
            SearchQueriesExplorer, **{"response_format": SearchQueriesStructure}
        )
        self.engine = engine

    def forward(self, topic: str, summary: str):
        with dspy.settings.context(lm=self.engine):
            search_queries_list = self.search_queries(
                topic=topic, summary=summary
            ).search_queries
            search_queries_list = json.loads(search_queries_list)["search_queries_list"]

        return dspy.Prediction(search_queries_list=search_queries_list)


class Summarizer(dspy.Signature):
    """atomic facts -> one sentence summary
    Guideline:
    1. The summary should be based on the atomic facts provided.
    2. The summary should be concise and to the point.
    """

    topic = dspy.InputField(prefix="Topic: ", format=str)
    fact_list = dspy.InputField(prefix="Atomic facts: ", format=list[str])
    summary = dspy.OutputField(prefix="Summary: ", format=str)


class Summarize(dspy.Module):
    """Summarize a topic given atomic facts."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.summarize = dspy.Predict(Summarizer)
        self.engine = engine

    def forward(self, topic: str, fact_list: list[str]):
        with dspy.settings.context(lm=self.engine):
            summary = self.summarize(topic=topic, fact_list=fact_list).summary

        return dspy.Prediction(summary=summary)
