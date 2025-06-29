import sys
import chromadb
from typing import List, Optional
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import dspy
from pydantic import BaseModel
from typing import Union


sys.path.append("..")

from utils import Parser, Logger
from config import Config

logger = Logger("memory")


class MemoryUnit:
    def __init__(self, uuid: str, content: str, url: str):
        self.uuid = uuid
        self.content = content
        self.url = url

    def to_dict(self):
        return {"uuid": self.uuid, "content": self.content, "url": self.url}

    def __repr__(self) -> str:
        return f"{self.content}"


class Memory:
    def __init__(
        self,
        topic: str,
        config: Config,
        engine: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None,
        force_recreate: bool = False,
    ):
        super().__init__()
        self.topic = topic
        self.config = config
        self.engine = engine

        self.chroma_client = chromadb.PersistentClient(config.DATABASE_DIR)

        collection_name = Parser.safe_title(self.topic)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        if force_recreate:
            self.chroma_client.delete_collection(collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )

    def add_information(
        self,
        memory_unit: MemoryUnit,
        custom_metadata: Optional[dict] = None,
        skip_similar: bool = True,
    ):
        query_result = self.collection.query(
            query_texts=[memory_unit.content], n_results=1
        )
        if (skip_similar and query_result["ids"] and len(query_result["ids"][0]) > 0 
            and query_result["documents"] and query_result["distances"]):
            content, distance = (
                query_result["documents"][0][0],
                query_result["distances"][0][0],
            )
            if distance <= float(self.config.SIMILARITY_THRESHOLD):
                logger.info(f'"{memory_unit.content}" is similar to "{content}"')
                return

        metadatas = {"url": memory_unit.url}
        if custom_metadata:
            metadatas.update(custom_metadata)

        self.collection.add(
            documents=[memory_unit.content],
            ids=[memory_unit.uuid],
            metadatas=[metadatas],
        )
        logger.info(f'"{memory_unit.content}" added to the memory.')

    def add_information_in_batch(
        self,
        memory_unit_list: List[MemoryUnit],
        custom_metadata: Optional[dict] = None,
        skip_similar: bool = True,
    ):
        for memory_unit in memory_unit_list:
            self.add_information(
                memory_unit=memory_unit,
                custom_metadata=custom_metadata,
                skip_similar=skip_similar,
            )

    def remove_information(self, uuid: str):
        self.collection.delete(ids=[uuid])
        logger.info(f'"{uuid}" removed from the memory.')

    def update_information(self, constraint: dict, new_metadata: dict):
        batch = self.collection.get(where=constraint)
        if not batch["metadatas"]:
            return
        updated_metadatas = []
        for metadata in batch["metadatas"]:
            metadata_dict = dict(metadata) if metadata else {}
            metadata_dict.update(new_metadata)
            updated_metadatas.append(metadata_dict)
        self.collection.update(ids=batch["ids"], metadatas=updated_metadatas)

    def count_information(self, constraint: Optional[dict] = None):
        if constraint is None:
            return self.collection.count()
        else:
            matching_items = self.collection.get(where=constraint)
            return len(matching_items["ids"])

    def retrieve_information(
        self, query: str, k: int = 5, constraint: Optional[dict] = None
    ) -> List[MemoryUnit]:
        query_result = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=constraint,
        )
        logger.info(
            f"Retrieved {len(query_result['ids'][0])} information for \"{query}\"."
        )
        memory_unit_list = []
        if (query_result["documents"] and query_result["metadatas"] and 
            query_result["ids"] and len(query_result["ids"][0]) > 0):
            for idx in range(len(query_result["ids"][0])):
                memory_unit = MemoryUnit(
                    uuid=query_result["ids"][0][idx],
                    content=query_result["documents"][0][idx],
                    url=str(query_result["metadatas"][0][idx]["url"]),
                )
                memory_unit_list.append(memory_unit)

        return memory_unit_list

    def group_information(
        self,
        n_clusters: Optional[int] = None,
        constraint: Optional[dict] = None,
        skip_small_cluster: bool = True,
    ):
        # Retrieve all documents and their embeddings
        memory_unit_list = self.collection.get(
            include=["embeddings", "documents"], where=constraint
        )
        embeddings = memory_unit_list["embeddings"]

        # Convert embeddings to numpy array and normalize
        embeddings_array = np.array(embeddings)
        embeddings_normalized = normalize(embeddings_array)

        if n_clusters is None:
            n_clusters = min(len(memory_unit_list["ids"]) // 10, 50)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_normalized)

        # Group documents by cluster
        clusters = defaultdict(list)
        if not memory_unit_list["documents"]:
            return clusters
        for i, memory_content in enumerate(memory_unit_list["documents"]):
            clusters[cluster_labels[i]].append(memory_content)

        # Remove clusters with less than 3 facts
        if skip_small_cluster == True:
            clusters = {k: v for k, v in clusters.items() if len(v) >= self.config.MIN_MEMORY_UNIT_PER_CLUSTER}

        # Sort clusters by their keys
        clusters = dict(sorted(clusters.items()))

        # Log cluster information
        for cluster_id in clusters:
            cluster_content_list = clusters[cluster_id]
            cluster_size = len(cluster_content_list)
            message = f"\nCluster {cluster_id}:\n"
            message += f"Size: {cluster_size}\n"
            message += "Sample Facts:\n"
            for content in cluster_content_list[
                :5
            ]:  # Log first 5 facts in each cluster
                message += f"- {content[:100]}...\n"
            # logger.info(message)

        return clusters

    def label_information(
        self, parent_path: str, label_list: list[str], constraint: Optional[dict] = None
    ):
        # Check if label_list is empty
        if not label_list:
            logger.info("Empty label list provided, returning empty list")
            return []

        memory_unit_list = self.collection.get(
            where=constraint, include=["documents", "embeddings", "metadatas"]
        )

        # Embed labels
        if not self.collection._embedding_function:
            logger.error("Embedding function not available")
            return []
        label_embeddings = self.collection._embedding_function(label_list)

        # First pass: Initial allocation
        allocated_tags = []
        label_counts = {}
        if not memory_unit_list["embeddings"]:
            logger.error("No embeddings available")
            return []
        for doc_embedding in memory_unit_list["embeddings"]:
            similarities = [
                np.dot(doc_embedding, label_emb) for label_emb in label_embeddings
            ]
            sorted_indices = np.argsort(similarities)[::-1]
            allocated_tags.append([label_list[i] for i in sorted_indices])
            primary_tag = allocated_tags[-1][0]
            label_counts[primary_tag] = label_counts.get(primary_tag, 0) + 1

        # Identify labels to keep
        labels_to_keep = [
            label for label in label_list if label_counts.get(label, 0) >= self.config.MIN_MEMORY_UNIT_PER_SECTION
        ]

        if len(labels_to_keep) == 1:
            logger.info(
                f"Only one label to keep: {labels_to_keep[0]}, discarding all subsection."
            )
            return []

        labels_to_discard = set(label_list) - set(labels_to_keep)

        # Second pass: Reallocate facts from discarded labels
        if not memory_unit_list["metadatas"]:
            logger.error("No metadata available")
            return []
        updated_metadatas = []
        for tags, metadata in zip(allocated_tags, memory_unit_list["metadatas"]):
            tag = next((tag for tag in tags if tag in labels_to_keep), tags[0])
            metadata_dict = dict(metadata) if metadata else {}
            metadata_dict["tag"] = f"{parent_path}//{tag}" if parent_path else tag
            updated_metadatas.append(metadata_dict)

        # Update the collection
        self.collection.update(
            ids=memory_unit_list["ids"],
            metadatas=updated_metadatas,
        )

        # Log the final memory unit counts
        for label in labels_to_keep:
            unit_count = len(
                self.collection.get(where={"tag": f"{parent_path}//{label}"})["ids"]
            )
            logger.info(f"Section: {label}, Memory Unit Count: {unit_count}")

        # Log discarded labels
        for label in labels_to_discard:
            logger.info(f"Discarded section due to insufficient memory units: {label}")

        return labels_to_keep
