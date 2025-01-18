import hashlib
from functools import wraps
import time
import logging
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def setup_logger():
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'remo.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} took {execution_time:.2f} seconds to execute")
        return result
    return wrapper

class Parser:
    @staticmethod
    def to_hash(text: str) -> str:
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int) -> List[str]:
        content_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )
        chunks = content_splitter.split_text(text)
        return chunks
    
    @staticmethod
    def clean_section_name(section_name: str) -> str:
        """Remove the number and dot from the section name"""
        return re.sub(r"^\d+\.\s*", "", section_name)
    
    @staticmethod
    def split_chinese_sentences(text: str) -> List[str]:
        # adapt from https://www.cnblogs.com/ting1/p/16833884.html
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", text) 
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para) 
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para) 
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        para = para.rstrip() 
        return para.split("\n")