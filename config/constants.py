from typing import Final
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class Constants:
    MAX_WEBPAGE: Final[int] = 3
    MAX_QUERY: Final[int] = 2
    MAX_SUBTOPIC: Final[int] = 3
    MAX_SEARCH_DEPTH: Final[int] = 2
    MAX_SUBTOPIC_EXPLORER_DEPTH: Final[int] = 1
    MAX_OUTLINE_DEPTH: Final[int] = 2
    MIN_MEMORY_UNITS_FOR_SUBSECTION: Final[int] = 10
    MAX_RESULTS: Final[int] = 10
    REASONING_MODEL: Final[str] = "openai/deepseek-r1"
    WRITING_MODEL: Final[str] = "openai/qwen-max"
    MODEL: Final[str] = "openai/deepseek-v3"

class FileType(Enum):
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    
class EmbeddingModel(Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    CONAN = "TencentBAC/Conan-embedding-v1"
    SONAR = "sonar"
