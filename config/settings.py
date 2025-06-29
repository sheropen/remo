import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # LM
        self.MODEL_NAME = "gpt-4o-mini-2024-07-18"
        self.BETTER_MODEL_NAME = "gpt-4o-2024-08-06"
        self.MAX_TOKENS = 4000
        
        # PATH
        self.DATABASE_DIR = "data/chroma"
        self.SERP_DIR = "data/serp"
        self.WEBPAGE_DIR = "data/webpage"
        self.OUTPUT_DIR = "data/output"
        self.LOG_DIR = "log"
        
        # Retriever
        self.MAX_WEBPAGE = 3
        self.MAX_QUERY = 2
        self.MAX_SUBTOPIC = 3
        self.MAX_SEARCH_DEPTH = 2

        # Planner
        self.MIN_MEMORY_UNIT_TO_EXPLORE = 10
        self.MAX_WRITE_DEPTH = 2
        self.MAX_CLUSTER_SUMMARIZE_MEMORY_UNIT = 300

        # Writer
        self.MIN_MEMORY_UNIT_FOR_WRITING = 5

        # Memory
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.MIN_MEMORY_UNIT_PER_CLUSTER = 3
        self.MIN_MEMORY_UNIT_PER_SECTION = 5
        self.SIMILARITY_THRESHOLD = 0.05

        # CONSTANTS
        self.EXCLUDE_DOMAIN_LIST = ["wikipedia.org", "youtube.com", ".pdf"]
        self.EXCLUDE_SECTION_LIST = [
            "introduction",
            "overview",
            "see also",
            "references",
            "external links",
            "further reading",
            "footnotes",
            "notes",
            "bibliography",
            "citations",
            "gallery",
            "sources",
            "additional information",
            "supplementary materials",
            "conclusion",
            "appendix",
        ]

        self._create_directories()
        load_dotenv()

    def _create_directories(self):
        """Create directories for all directory paths"""
        dir_attrs = [attr for attr in dir(self) if attr.endswith("_DIR")]
        for attr_name in dir_attrs:
            dir_path = getattr(self, attr_name)
            if isinstance(dir_path, str):
                os.makedirs(dir_path, exist_ok=True)



 