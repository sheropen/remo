from collections import defaultdict
from typing import List
from memory import MemoryUnit
import json
import os
from nltk.tokenize import sent_tokenize
from bidict import bidict
import re
from utils import Parser


class Outline:
    def __init__(self, title: str, layer: int = 0):
        self.title: str = Parser.parse_section_name(title)
        self.layer: int = layer
        self.children: List["Outline"] = []

    def create_child(self, title: str) -> "Outline":
        child = Outline(title=title, layer=self.layer + 1)
        self.children.append(child)
        return child

    def insert_child(self, child: "Outline"):
        child.layer = self.layer + 1
        child.update_children_level()
        self.children.append(child)

    def update_children_level(self):
        level = self.layer + 1
        for child in self.children:
            child.layer = level
            child.update_children_level()

    def to_markdown(self) -> str:
        if self.layer == 0:  # the article title
            markdown = f"=={self.title}==\n"
        else:
            markdown = f"{'#' * self.layer} {self.title}\n"
        for child in self.children:
            markdown += child.to_markdown()
        return markdown

    def to_flatten_list(self, parent_title: str = "") -> List[str]:
        flat_list = []
        if self.layer == 0:
            full_title = self.title
        else:
            full_title = f"{parent_title}//{self.title}"
            flat_list.append(full_title)
        for child in self.children:
            flat_list.extend(child.to_flatten_list(parent_title=full_title))
        return flat_list

    def to_dict(self):
        outline_dict = {
            "section_title": self.title,
            "subsection_list": [],
        }
        for child in self.children:
            outline_dict["subsection_list"].append(child.to_dict())
        return outline_dict

    @classmethod
    def from_dict(cls, data: dict):
        outline = cls(title=data["section_title"])
        for subsection_data in data["subsection_list"]:
            child = cls.from_dict(subsection_data)
            outline.insert_child(child)
        if len(outline.children) == 1:
            outline.children = outline.children[0].children
        return outline

    @classmethod
    def from_flatten_list(cls, flat_list: List[str]):
        if not flat_list:
            return None

        root_title = flat_list[0].split("//")[0]
        root = cls(title=root_title)

        for path in flat_list:
            sections = path.split("//")
            current = root
            for section in sections[1:]:  # Skip the root title
                child = next((c for c in current.children if c.title == section), None)
                if not child:
                    child = current.create_child(section)
                current = child
        return root

    def __repr__(self) -> str:
        return self.to_markdown()


# Article
class Sentence:
    def __init__(
        self,
        content: str,
        proposition: str = None,
        citation_list: List[MemoryUnit] = [],
        doc_id_list: List[int] = [],
        keep_citation_numbers: bool = False,
    ):
        if keep_citation_numbers:
            self.content = content
        else:
            self.content = Parser.remove_citation_number(content)
        self.proposition = proposition
        self.citation_list = citation_list
        self.doc_id_list = doc_id_list

    def __repr__(self, show_citation_numbers=True) -> str:
        """Returns string representation of sentence with optional citation numbers"""
        if self.doc_id_list is None:
            return self.content

        result = Parser.remove_citation_number(self.content).strip()

        if show_citation_numbers and self.doc_id_list:
            unique_doc_ids = sorted(set(self.doc_id_list))

            result += "["
            for doc_id in unique_doc_ids:
                result += f"{doc_id},"
            result = result[:-1] + "]"

        return result

    def to_dict(self):
        return {
            "content": Parser.remove_citation_number(self.content),
            "citation_list": [citation.to_dict() for citation in self.citation_list],
            "proposition": self.proposition,
        }


class Paragraph:
    def __init__(self):
        self.sentence_list: List[Sentence] = []

    def add_sentence(self, sentence: Sentence):
        self.sentence_list.append(sentence)

    def __repr__(self, show_citation_numbers=True) -> str:
        return " ".join(
            sentence.__repr__(show_citation_numbers=show_citation_numbers)
            for sentence in self.sentence_list
        )

    def to_dict(self):
        return {
            "sentence_list": [sentence.to_dict() for sentence in self.sentence_list]
        }


class Article:
    """A class representing a hierarchical article structure with sections, subsections, and paragraphs."""

    def __init__(
        self, title: str, layer: int = 0, working_context: List[MemoryUnit] = []
    ):
        # Basic article properties
        self.title = Parser.parse_section_name(title.split("//")[-1])
        self.layer = layer
        self.working_context = working_context

        # Content structure
        self.subsection_list: List[Article] = []
        self.paragraph_list: List[Paragraph] = []

        # Citation tracking
        self.doc_id_url_map = bidict()
        self.cited_mu_by_doc_id = defaultdict(list)
        self.current_doc_id = 1

    def url_to_doc_id(self, url: str) -> int:
        """Maps a URL to a document ID, creating a new ID if needed."""
        if url not in self.doc_id_url_map.inverse:
            self.doc_id_url_map[self.current_doc_id] = url
            self.current_doc_id += 1
        return self.doc_id_url_map.inverse[url]

    def doc_id_to_url(self, doc_id: int) -> str:
        """Retrieves the URL associated with a document ID."""
        return self.doc_id_url_map[doc_id]

    def record_cited_mu(self, mu: MemoryUnit) -> int:
        """Records a cited memory unit and returns its document ID."""
        doc_id = self.url_to_doc_id(mu.url)
        self.cited_mu_by_doc_id[doc_id].append(mu)
        return doc_id

    def add_subsection(self, subsection: "Article"):
        """Adds a subsection to the article."""
        self.subsection_list.append(subsection)

    def add_paragraph(self, paragraph: Paragraph):
        """Adds a paragraph to the article."""
        self.paragraph_list.append(paragraph)

    def get_all_working_context(self) -> str:
        """Returns a formatted string of all working context across the article hierarchy."""
        result = ""
        if self.layer != 0:
            result += f"{'#' * self.layer} {self.title}\n"

        for mu in self.working_context:
            result += f" - {mu}\n"
        result += "\n"

        for subsection in self.subsection_list:
            result += subsection.get_all_working_context()
        return result

    def get_all_sentence(self) -> List[Sentence]:
        """Returns a flat list of all sentences in the article and its subsections."""
        sentence_list = []
        for paragraph in self.paragraph_list:
            sentence_list.extend(paragraph.sentence_list)
        for subsection in self.subsection_list:
            sentence_list.extend(subsection.get_all_sentence())
        return sentence_list

    @classmethod
    def from_text(
        cls,
        title: str,
        text: str,
        layer: int,
        working_context: List[MemoryUnit] = [],
        keep_citation_numbers: bool = False,
    ) -> "Article":
        """Creates an Article instance from formatted text."""
        article = cls(title=title, layer=layer, working_context=working_context)
        paragraph_list = text.strip().split("\n")
        paragraph_list = [para for para in paragraph_list if para.strip()]

        i = 0
        while i < len(paragraph_list):
            para_text = paragraph_list[i]

            # Handle title
            if para_text.strip().startswith(f"{'#' * layer} "):
                article.title = para_text.strip()[layer + 1 :]
                i += 1
                continue

            # Handle subsection
            elif para_text.strip().startswith(f"{'#' * (layer + 1)} "):
                subsection_title = para_text.strip()[layer + 2 :]
                subsection_content = []
                i += 1

                # Collect subsection content
                while i < len(paragraph_list):
                    next_para = paragraph_list[i]
                    if next_para.strip().startswith(f"{'#' * (layer + 1)} "):
                        break
                    subsection_content.append(next_para)
                    i += 1

                # Create and add subsection
                subsection_text = "\n\n".join(subsection_content)
                subsection = cls.from_text(
                    title=subsection_title,
                    text=subsection_text,
                    layer=layer + 1,
                    working_context=working_context,
                )
                article.add_subsection(subsection)
                continue
            elif para_text.strip() == "":
                i += 1
                continue
            # Handle regular paragraph
            else:
                paragraph = Paragraph()
                sentence_list = sent_tokenize(para_text)

                # Combine short sentences with previous ones
                processed_sentences = []
                for j, sentence_text in enumerate(sentence_list):
                    words = sentence_text.split()
                    if j > 0 and len(words) <= 3:
                        processed_sentences[-1] += " " + sentence_text
                    else:
                        processed_sentences.append(sentence_text)

                for sentence_text in processed_sentences:
                    paragraph.add_sentence(
                        Sentence(
                            content=sentence_text,
                            keep_citation_numbers=keep_citation_numbers,
                        )
                    )
                article.add_paragraph(paragraph)
                i += 1

        return article

    @classmethod
    def from_json(cls, file_path: str) -> "Article":
        """Creates an Article instance from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Article":
        """Creates an Article instance from a dictionary representation."""
        article = cls(title=data["title"], layer=data["layer"])

        # Handle references if present
        if "reference_list" in data:
            for doc_id, ref_data in data["reference_list"].items():
                doc_id = int(doc_id)
                article.doc_id_url_map[doc_id] = ref_data["url"]
                article.cited_mu_by_doc_id[doc_id] = [
                    MemoryUnit(**mu_data) for mu_data in ref_data["content"]
                ]

            article.current_doc_id = (
                max(article.doc_id_url_map.keys()) + 1 if article.doc_id_url_map else 1
            )

        # Load working context
        article.working_context = [
            MemoryUnit(**mu_data) for mu_data in data["working_context"]
        ]

        # Load paragraphs
        for para_data in data["paragraph_list"]:
            paragraph = Paragraph()
            for sent_data in para_data["sentence_list"]:
                sentence = Sentence(content=sent_data["content"])
                sentence.citation_list = []
                sentence.doc_id_list = []
                for citation in sent_data["citation_list"]:
                    sentence.citation_list.append(MemoryUnit(**citation))
                    sentence.doc_id_list.append(article.url_to_doc_id(citation["url"]))
                paragraph.add_sentence(sentence)
            article.add_paragraph(paragraph)

        # Load subsections recursively
        for subsection_data in data["subsection_list"]:
            subsection = cls.from_dict(subsection_data)
            article.add_subsection(subsection)

        return article

    def to_dict(self) -> dict:
        """Converts the article to a dictionary representation."""
        result = {
            "title": self.title,
            "layer": self.layer,
            "working_context": [mu.to_dict() for mu in self.working_context],
            "subsection_list": [
                subsection.to_dict() for subsection in self.subsection_list
            ],
            "paragraph_list": [
                paragraph.to_dict() for paragraph in self.paragraph_list
            ],
        }

        if self.layer == 0:
            result["reference_list"] = {
                doc_id: {
                    "url": url,
                    "content": [mu.to_dict() for mu in self.cited_mu_by_doc_id[doc_id]],
                }
                for doc_id, url in self.doc_id_url_map.items()
            }

        return result

    def save_file(self, output_dir: str):
        """Saves the article in multiple formats (raw_txt, json, clean_txt)."""

        def get_dir(format: str) -> str:
            output_format_dir = os.path.join(output_dir, format)
            if not os.path.exists(output_format_dir):
                os.makedirs(output_format_dir)
            return output_format_dir

        # Save as raw text
        with open(
            f"{get_dir('raw_txt')}/{Parser.safe_title(self.title)}.txt", "w"
        ) as f:
            f.write(self.__repr__())

        # Save as cleaned text
        with open(
            f"{get_dir('clean_txt')}/{Parser.safe_title(self.title)}.txt", "w"
        ) as f:
            f.write(self.__repr__(show_lead_section=False, show_citation_numbers=False))

        # Save as JSON
        with open(
            f"{get_dir('json')}/{Parser.safe_title(self.title)}.json",
            "w",
        ) as f:
            json.dump(self.to_dict(), f)

    def __repr__(
        self, show_citation=False, show_lead_section=True, show_citation_numbers=True
    ) -> str:
        """Returns a string representation of the article."""
        result = ""

        # Handle section title
        if self.layer != 0:
            title = self.title.split("//")[-1]
            result += f"{'#' * self.layer} {title}\n"

        # Handle lead section and paragraphs
        if show_lead_section or self.layer != 0:
            for paragraph in self.paragraph_list:
                result += f"{paragraph.__repr__(show_citation_numbers=show_citation_numbers)}\n\n"

        # Handle subsections
        for subsection in self.subsection_list:
            result += subsection.__repr__(
                show_citation=show_citation,
                show_lead_section=show_lead_section,
                show_citation_numbers=show_citation_numbers,
            )

        # Add references for root article
        if show_citation and self.layer == 0:
            result += "# References\n"
            for doc_id, mu_list in self.cited_mu_by_doc_id.items():
                url = self.doc_id_to_url(doc_id)
                result += f"[{doc_id}]{url}\n"
                for mu in mu_list:
                    result += f" - {mu}\n"
            result += "\n# Memory Unit Used\n"
            result += self.get_all_working_context()

        return result
