import os
import chromadb
from dataclasses import dataclass
from typing import List, Optional
import uuid
from sklearn.cluster import KMeans
import dspy
import json
import concurrent.futures
from collections import defaultdict
from typing import Union

from config.paths import DB_DIR
from config.constants import EmbeddingModel
from src.data_structure import MemoryUnit, Webpage
from src.utils import Parser, timer, setup_logger


logger = setup_logger()
MAX_SIMILARITY_THRESHOLD = 1 - 0.95


class SectionAssigner(dspy.Signature):
    """给定一个句子和章节列表，将句子分配到最合适的章节。
    输出应为输入列表中的其中一个章节。
    """

    sentence = dspy.InputField()
    section_list = dspy.InputField(desc="章节列表")
    answer = dspy.OutputField(desc="章节名称")


class Memory:
    def __init__(
        self,
        topic: str,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel] = None,
        force_recreate: bool = False,
    ):
        self.topic = topic
        self.engine = engine

        _dir = os.path.join(DB_DIR, "default")
        self.chroma_client = chromadb.PersistentClient(_dir)

        model_dir = os.getenv("MODEL_DIR")
        if model_dir:
            model_path = os.path.join(model_dir, EmbeddingModel.CONAN.value)
        else:
            model_path = EmbeddingModel.CONAN.value

        sentence_transformer_ef = (
            chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_path
            )
        )

        # chromadb collection name must be less than 64 characters
        collection_name = Parser.to_hash(self.topic)[:63]

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=sentence_transformer_ef
        )

        if force_recreate:
            self.chroma_client.delete_collection(collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name, embedding_function=sentence_transformer_ef
            )

        self.assigner = dspy.ChainOfThought(SectionAssigner)

    @timer
    def insert_information(self, memory_units: List[MemoryUnit]):
        metadatas = []
        for unit in memory_units:
            source = unit.source
            metadatas.append({"source": source.url})

        self.collection.add(
            documents=[unit.content for unit in memory_units],
            ids=[str(uuid.uuid4()) for _ in memory_units],
            metadatas=metadatas,
        )
        logger.info(f"{self.collection.count()} memory units inserted")

    @timer
    def deduplicate_information(self):
        results = self.collection.get(include=["documents"])
        query_results = self.collection.query(
            query_texts=results["documents"], n_results=2
        )  # Get 2 results to skip self-match

        id_to_content = {}

        similarity_groups = {}  # Maps unit_id to its group representative
        for idx, unit_id in enumerate(results["ids"]):
            id_to_content[unit_id] = results["documents"][idx]
            if unit_id in similarity_groups:
                continue
            similar_unit_id = query_results["ids"][idx][
                1
            ]  # Use second result to skip self
            similar_distance = query_results["distances"][idx][1]

            if similar_distance <= MAX_SIMILARITY_THRESHOLD:
                group_rep = similarity_groups.get(similar_unit_id, unit_id)
                similarity_groups[unit_id] = group_rep
                similarity_groups[similar_unit_id] = group_rep

        # print out all similar items within a group
        groups = defaultdict(list)
        for unit_id, group_rep in similarity_groups.items():
            groups[group_rep].append(id_to_content[unit_id])

        for group_rep, contents in groups.items():
            if len(contents) > 1:  # 只打印有重复项的组
                logger.info(f"Similar content group:")
                for content in contents:
                    logger.info(f"- {content[:100]}...")  # 只显示前100个字符
                logger.info("---")

        remove_ids = {
            unit_id
            for unit_id, group_rep in similarity_groups.items()
            if unit_id != group_rep
        }

        if remove_ids:
            self.collection.delete(ids=list(remove_ids))
            logger.info(
                f"{len(remove_ids)} memory units deleted out of {len(results['ids'])}, deduplication rate: {100 * len(remove_ids) / len(results['ids'])} %"
            )

    @timer
    def retrieve_information(
        self, query: str, k: int = 100, constraint: dict = None
    ) -> List[str]:
        query_result = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=constraint,
        )

        memory_unit_list = []
        for idx in range(len(query_result["ids"][0])):
            webpage = Webpage(url=query_result["metadatas"][0][idx]["source"])
            memory_unit = MemoryUnit(
                content=query_result["documents"][0][idx], source=webpage
            )
            memory_unit_list.append(memory_unit)
        return memory_unit_list

    @timer
    def group_information(
        self, constraint: dict = None, n_clusters: int = 20, query: str = None
    ):
        if query:
            query_result = self.collection.query(
                query_texts=[query],
                n_results=100,
                where=constraint,
                include=["embeddings", "documents"],
            )
            documents = query_result["documents"][0]
            embeddings = query_result["embeddings"][0]
        else:
            query_result = self.collection.get(
                where=constraint, include=["embeddings", "documents"]
            )
            documents = query_result["documents"]
            embeddings = query_result["embeddings"]

        if not documents:
            return []

        # Perform k-means clustering
        memory_unit_count = len(documents)
        cluster_count = max(n_clusters, memory_unit_count // 5)

        kmeans = KMeans(n_clusters=min(n_clusters, len(documents)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Group documents by cluster
        clustered_docs = [[] for _ in range(max(clusters) + 1)]
        for doc, cluster_id in zip(documents, clusters):
            clustered_docs[cluster_id].append(doc)

        return clustered_docs

    @timer
    def label_information(self, labels: List[str], constraint: dict = None):
        """Label information with given labels"""

        # Get memory units matching constraint
        memory_unit_list = (
            self.collection.get(where=constraint)
            if constraint
            else self.collection.get()
        )
        if not memory_unit_list["ids"]:
            return []

        def assign_label(content, _label_list):
            """Assign a label to content using LLM"""
            with dspy.settings.context(lm=self.engine):
                result = self.assigner(
                    sentence=content, section_list=",".join(_label_list)
                )
                try:
                    label = result.answer
                    return label if label in _label_list else "Other"
                except:
                    logger.error(f"Failed to parse result: {result.answer}")
                    return "Other"

        # Assign initial labels using parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            allocated_labels = list(
                executor.map(
                    lambda x: assign_label(x, labels), memory_unit_list["documents"]
                )
            )

        # Count valid labels
        label_counts = defaultdict(int)
        for label in allocated_labels:
            if label != "Other":
                label_counts[label] += 1

        # Keep labels with >= 5 memory units
        labels_to_keep = [label for label in labels if label_counts.get(label, 0) >= 5]
        if len(labels_to_keep) <= 1:
            logger.info("Not enough labels to keep - discarding all subsections")
            return []

        # Reassign discarded labels
        contents_to_reassign = [
            content
            for content, label in zip(memory_unit_list["documents"], allocated_labels)
            if label not in labels_to_keep
        ]

        new_labels = []
        if contents_to_reassign:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                new_labels = list(
                    executor.map(
                        lambda x: assign_label(x, labels_to_keep), contents_to_reassign
                    )
                )

        # Update metadata with final labels
        label_idx = 0
        for old_label, metadata in zip(allocated_labels, memory_unit_list["metadatas"]):
            label = old_label if old_label in labels_to_keep else new_labels[label_idx]
            if old_label not in labels_to_keep:
                label_idx += 1
            metadata["label"] = label

        # Update collection
        self.collection.update(
            ids=memory_unit_list["ids"], metadatas=memory_unit_list["metadatas"]
        )

        # Log results
        discarded = set(labels) - set(labels_to_keep)
        for label in discarded:
            logger.info(f"Discarded section due to insufficient memory units: {label}")

        for label in labels_to_keep:
            unit_count = len(self.collection.get(where={"label": label})["ids"])
            logger.info(f"Section: {label}, Memory Unit Count: {unit_count}")

        return labels_to_keep
