from pydantic import BaseModel
from typing import List


class Webpage:
    def __init__(
        self,
        url: str,
        content: str = None,
        title: str = None,
        snippet: str = None,
        date: str = None,
        query: str = None,
    ):
        self.url = url
        self.content = content
        self.title = title
        self.snippet = snippet
        self.date = date
        self.query = query

    def to_dict(self):
        return {
            "url": self.url,
            "content": self.content,
            "title": self.title,
            "snippet": self.snippet,
            "date": self.date,
            "query": self.query,
        }


class MemoryUnit:
    def __init__(
        self,
        content: str,
        context: str = None,
        topic: str = None,
        source: Webpage = None,
    ):
        self.content = content
        self.context = context
        self.topic = topic
        self.source = source

    def to_dict(self):
        return {
            "content": self.content,
            "context": self.context,
            "topic": self.topic,
            "source": self.source.to_dict(),
        }


class Information(BaseModel):
    entity: str
    context: str


class Sentence:
    def __init__(self, content: str, citation_list: List[MemoryUnit] = None):
        self.content = content
        if citation_list is None:
            self.citation_list = []
        else:
            self.citation_list = citation_list

    def __repr__(self) -> str:
        return self.content

    def to_dict(self):
        return {
            "content": self.content,
            "citation_list": [citation.to_dict() for citation in self.citation_list],
        }


class Paragraph:
    def __init__(self, sentence_list: List[Sentence] = None):
        if sentence_list is None:
            self.sentence_list = []
        else:
            self.sentence_list = sentence_list

    def add_sentence(self, sentence: Sentence):
        self.sentence_list.append(sentence)

    def to_dict(self):
        return {
            "sentence_list": [sentence.to_dict() for sentence in self.sentence_list]
        }

    def __repr__(self) -> str:
        return " ".join(sentence.__repr__() for sentence in self.sentence_list)
