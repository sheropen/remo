from bidict import bidict
import json
from pathlib import Path
from typing import List
from src.utils import Parser
from src.data_structure import MemoryUnit, Sentence, Paragraph

class Outline:
    def __init__(self, title: str, layer: int = 1):
        self.title: str = Parser.clean_section_name(title)
        self.layer: int = layer
        self.children: List["Outline"] = []

    def insert_child(self, child: "Outline"):
        child.layer = self.layer + 1
        child.update_children_level()
        self.children.append(child)

    def update_children_level(self):
        level = self.layer + 1
        for child in self.children:
            child.layer = level
            child.update_children_level()

    def to_markdown(self, show_title: bool = False) -> str:
        if self.layer != 1 or show_title:  # the article title
            markdown = f"{'#' * self.layer} {self.title.split('//')[-1]}\n"
        else:
            markdown = ""
        for child in self.children:
            markdown += child.to_markdown()
        return markdown

    def to_flatten_list(self, only_leaf: bool = False) -> List[str]:
        flat_list = []
        if self.layer != 1: # not root node
            if only_leaf == False:
                flat_list.append(self.title)    
            elif len(self.children) == 0:
                flat_list.append(self.title)
        for child in self.children:
            flat_list.extend(child.to_flatten_list())
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
    def from_markdown(cls, title: str, markdown: str):
        lines = markdown.strip().split('\n')
        root = cls(title=title)
        current_levels = [root]
        current_titles = [title]  # Track full hierarchical titles
        base_level = None
        
        for line in lines:
            if not line.startswith('#'):
                continue
                
            # Count number of # to determine level
            level = 0
            while level < len(line) and line[level] == '#':
                level += 1
            
            # Set base level on first heading if not set
            if base_level is None:
                base_level = level
            
            # Normalize level relative to base level
            normalized_level = level - base_level + 1
                
            raw_title = line[level:].strip()
            
            # Handle nesting
            while len(current_levels) > normalized_level:
                current_levels.pop()
                current_titles.pop()
            
            # Ensure we have a parent to attach to
            if not current_levels:
                current_levels.append(root)
                current_titles.append(title)
                
            # Create full hierarchical title path
            full_title = "//".join(current_titles + [raw_title])
            
            # Create new outline node
            node = Outline(title=full_title, layer=normalized_level)
            
            # Attach to parent and update tracking lists
            current_levels[-1].insert_child(node)
            current_levels.append(node)
            current_titles.append(raw_title)
            
        return root

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
    

class Article:
    def __init__(
        self, title: str, layer: int = 1, working_context: List[MemoryUnit] = []
    ):
        self.title = title
        self.layer = layer
        self.working_context = working_context

        # Content structure
        self.subsection_list: List[Article] = []
        self.paragraph_list: List[Paragraph] = []

        # Citation tracking
        self.reference_dict = bidict()
        self.doc_cnt = 1

    def add_reference(self, url: str):
        if url not in self.reference_dict.inverse:
            self.reference_dict[self.doc_cnt] = url
            self.doc_cnt += 1

    def add_subsection(self, subsection: "Article"):
        self.subsection_list.append(subsection)

    def add_paragraph(self, paragraph: Paragraph):
        self.paragraph_list.append(paragraph)
    
    def get_all_working_context(self):
        working_context = self.working_context
        for subsection in self.subsection_list:
            working_context.extend(subsection.get_all_working_context())
        return working_context
    
    
    def update_reference_dict(self):
        for working_context in self.get_all_working_context():
            self.add_reference(working_context.source.url)
        
    @classmethod
    def from_text(cls, title: str, layer: int, text: str, working_context: List[MemoryUnit] = []):
        article = cls(title=title, layer=layer, working_context=working_context)
        paragraph_list = text.strip().split('\n\n')
        for paragraph in paragraph_list:
            sentence_list = Parser.split_chinese_sentences(paragraph)
            sentence_list = [Sentence(content=sentence) for sentence in sentence_list if sentence.strip()]
            article.add_paragraph(Paragraph(sentence_list=sentence_list))
        return article

    def to_dict(self):
        result = {
            "title": self.title,
            "layer": self.layer,
            "working_context": [working_context.to_dict() for working_context in self.working_context],
            "subsection_list": [subsection.to_dict() for subsection in self.subsection_list],
            "paragraph_list": [paragraph.to_dict() for paragraph in self.paragraph_list],
        }
        if self.layer == 1:
            result["reference_dict"] = {k: v for k, v in self.reference_dict.items()}
        return result
        
    def save_to_files(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_dir = output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        with open(json_dir / f"{self.title}.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            
        markdown_dir = output_dir / "markdown"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        with open(markdown_dir / f"{self.title}.md", "w", encoding="utf-8") as f:
            f.write(self.__repr__(show_citation=True, show_reference=True))
            
        txt_dir = output_dir / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        with open(txt_dir / f"{self.title}.txt", "w", encoding="utf-8") as f:
            f.write(self.__repr__(show_citation=False, show_reference=True))
            
    def __repr__(self, show_citation=False, show_reference=True) -> str:
        text = ""
        text += f"{'#' * self.layer} {self.title.split('//')[-1]}\n"
        for paragraph in self.paragraph_list:
            for sentence in paragraph.sentence_list:
                if show_citation:
                    cited_doc_id_set = set()
                    for citation in sentence.citation_list:
                        cited_doc_id_set.add(self.reference_dict.inverse[citation.source.url])
                    cited_doc_ids = sorted(list(cited_doc_id_set))
                    text += f"{sentence.content}[{','.join(cited_doc_ids)}]"
                else:
                    text += f"{sentence.content}"
            text += "\n\n"
        for subsection in self.subsection_list:
            subsection_text = subsection.__repr__(show_citation=show_citation)
            if subsection_text.strip():
                text += f"{subsection_text}"
        
        if show_reference and self.layer == 1:
            text += "## 参考文献\n"
            for doc_id in range(1, self.doc_cnt):
                text += f"[{doc_id}] {self.reference_dict[doc_id]}\n"
        return text
