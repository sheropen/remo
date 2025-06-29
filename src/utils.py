import re
import os
import sys
import logging
import toml
from typing import List, Dict, TYPE_CHECKING, Optional
import httpx
import concurrent.futures
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import extract
from nltk.tokenize import sent_tokenize, word_tokenize
from pypinyin import lazy_pinyin

sys.path.append(".")

from config import Config

class Logger:
    def __init__(
        self,
        name,
        level=logging.INFO,
    ):
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.log_file = os.path.join(os.getcwd(), f"log/{name}.log")
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.logger.addHandler(self.create_file_handler())
        self.logger.addHandler(self.create_console_handler())

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def create_file_handler(self):
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(self.level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        return fh

    def create_console_handler(self):
        ch = logging.StreamHandler()
        ch.setLevel(self.level)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        return ch





class Parser:
    @staticmethod
    def to_hash(text: str) -> str:
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        return hash_obj.hexdigest()
    
    @staticmethod
    def safe_title(title: str) -> str:
        """Sanitize a title string to be safe for use as a collection/file name."""

        # Convert Chinese characters to pinyin while preserving English
        parts = []
        current_part = []

        for char in title:
            if "\u4e00" <= char <= "\u9fff":  # Chinese character range
                if current_part:
                    parts.append("".join(current_part))
                    current_part = []
                parts.extend(lazy_pinyin(char))
            else:
                current_part.append(char)

        if current_part:
            parts.append("".join(current_part))

        pinyin_str = "_".join(parts)

        # Replace any non-alphanumeric chars (except underscore) with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", pinyin_str)

        # Remove any leading/trailing underscores
        sanitized = sanitized.strip("_")

        # maximum 60 characters for collection name
        return sanitized[:60]

    @staticmethod
    def split_doc_into_chunks(document, max_chunk_len=1000):
        """Split document into chunks of approximately max_chunk_len words"""
        chunks = []
        current_chunk = []
        words_in_chunk = 0

        for sentence in sent_tokenize(document):
            words = word_tokenize(sentence)
            word_count = len(words)

            # Handle sentences longer than max length by splitting them
            if word_count > max_chunk_len:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    words_in_chunk = 0

                # Split long sentence into chunks
                for i in range(0, word_count, max_chunk_len):
                    chunks.append(" ".join(words[i : i + max_chunk_len]))
                continue

            # Start new chunk if current would exceed max length
            if words_in_chunk + word_count > max_chunk_len:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                words_in_chunk = word_count
            else:
                current_chunk.append(sentence)
                words_in_chunk += word_count

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def remove_citation_number(text: str) -> str:
        return re.sub(r"\s*\[\d+\]", "", text)

    @staticmethod
    def parse_section_name(section_name: str) -> str:
        """Remove the number and dot from the section name"""
        return re.sub(r"^\d+\.\s*", "", section_name)


class WebsiteContentProcessor:
    """Process and extract content from web pages.

    Acknowledgement: This part of code is adapted from https://github.com/stanford-oval/storm.
    """

    def __init__(
        self,
        min_content_length: int = 150,
        content_chunk_size: int = 1000,
        max_concurrent_requests: int = 10,
        config: Optional["Config"] = None,
    ):
        """
        Args:
            min_content_length: Minimum character count for the article to be considered valid.
            content_chunk_size: Maximum character count for each content chunk.
            max_concurrent_requests: Maximum number of concurrent requests for downloading webpages.
        """
        if config is None:
            from config import Config
            config = Config()
        self.config = config
        self.http_client = httpx.Client(verify=False)
        self.min_content_length = min_content_length
        self.max_concurrent_requests = max_concurrent_requests
        self.content_splitter = RecursiveCharacterTextSplitter(
            chunk_size=content_chunk_size,
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

    def fetch_webpage(self, url: str):
        cached_page_path = os.path.join(
            self.config.WEBPAGE_DIR, f"{Parser.to_hash(url)}.html"
        )
        if os.path.exists(cached_page_path):
            with open(cached_page_path, "r") as f:
                return f.read()
        try:
            response = self.http_client.get(url, timeout=4)
            if response.status_code >= 400:
                response.raise_for_status()
            with open(cached_page_path, "w") as f:
                f.write(response.text)
            return response.text
        except httpx.HTTPError as error:
            print(f"Error while requesting {error.request.url!r} - {error!r}")
            return None

    def extract_articles(self, urls: List[str]) -> Dict:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_concurrent_requests
        ) as executor:
            webpage_contents = list(executor.map(self.fetch_webpage, urls))

        extracted_articles = {}

        for content, url in zip(webpage_contents, urls):
            if content is None:
                continue
            try:
                article_text = extract(
                    content,
                    include_tables=False,
                    include_comments=False,
                    output_format="text",
                )
            except Exception as e:
                print(f"Error extracting article from {url}: {e}")
                continue
            if article_text is not None and len(article_text) > self.min_content_length:
                extracted_articles[url] = {"text": article_text}

        return extracted_articles

    def extract_content_chunks(self, urls: List[str]) -> Dict:
        articles = self.extract_articles(urls)
        for url in articles:
            articles[url]["snippets"] = self.content_splitter.split_text(
                articles[url]["text"]
            )

        return articles
