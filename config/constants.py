from typing import Final
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class RetrieverConfig:
    MAX_WEBPAGE: Final[int] = 3
    MAX_QUERY: Final[int] = 2
    MAX_SUBTOPIC: Final[int] = 3
    MAX_SEARCH_DEPTH: Final[int] = 2

# File types
class FileType(Enum):
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    
class EmbeddingModel(Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    CONAN = "TencentBAC/Conan-embedding-v1"
    SONAR = "sonar"
