import dspy
from typing import List
from pydantic import BaseModel

from src.utils import timer
from src.data_structure import MemoryUnit


class Information(BaseModel):
    id: int
    content: str


class Writer:
    def __init__(self, engine: dspy.dsp.LM, better_engine: dspy.dsp.LM):
        self.engine = engine
        self.better_engine = better_engine

    @timer
    def write_article(self, topic: str, focus: str, information: List[MemoryUnit]):
        f = dspy.Predict(ArticleWriter)
        information = ""
        for idx, unit in enumerate(information):
            information += f"{idx+1}. {unit.content}\n"
        with dspy.settings.context(lm=self.engine):
            response = f(topic=topic, focus=focus, information=information)
        return response.article

    @timer
    def write_section(
        self,
        topic: str,
        summary: str,
        focus: str,
        information: List[str],
        section_title: str,
    ):
        f = dspy.Predict(SectionWriter)
        with dspy.settings.context(lm=self.engine):
            response = f(
                topic=topic,
                summary=summary,
                focus=focus,
                information=information,
                section_title=section_title,
            )
        return response.section

    @timer
    def summarize_information(self, information: List[str]):
        f = dspy.Predict(Summarizer)
        response = f(information=information)
        return response.summary

    @timer
    def rewrite_article(self, article: str):
        f = dspy.Predict(ArticleRewriter, max_tokens=8000)
        with dspy.settings.context(lm=self.better_engine):
            response = f(article=article)
        return response.rewritten_article

    @timer
    def find_citations(self, sentence: str, information_list: List[MemoryUnit]):
        information = [
            Information(id=idx, content=unit.content)
            for idx, unit in enumerate(information_list)
        ]

        f = dspy.Predict(CitationAdder)
        with dspy.settings.context(lm=self.engine):
            response = f(sentence=sentence, information=information)
        return response.cited_indices


class ArticleRewriter(dspy.Signature):
    """
    改写百科文章，使得整体（句子之间、不同章节之间）连贯且易懂。
    # 生成规则：
    1. 去除冗余的信息，确保整体流畅，在每章的开头不必重复同样的内容
    2. 以markdown格式返回
    3. 必须使用中文输出
    """

    article = dspy.InputField(prefix="文章：")
    rewritten_article = dspy.OutputField(prefix="改写后的文章：")


class ArticleWriter(dspy.Signature):
    """
    给定主题、聚焦点、信息，以维基百科风格输出文章
    # 生成规则
    1. 使用markdown格式输出，# 标题，## 小标题，### 小标题
    2. 将文章分成多个章节，每个章节包含一个或多个段落
    3. 不要引用任何信息来源
    4. 必须使用中文输出
    """

    topic = dspy.InputField(prefix="主题：")
    focus = dspy.InputField(prefix="聚焦点：")
    information = dspy.InputField(prefix="信息：")
    article = dspy.OutputField(prefix="文章：")


class SectionWriter(dspy.Signature):
    """
    给定主题、聚焦点、信息，以及章节标题，生成章节内容
    生成规则
    1. 不要在输出中包含章节名称。
    2. 只输出与当前章节最相关的信息。
    3. 所生成内容是完整百科的其中一个部分，所以不必重复一些较笼统的信息。
    4. 不要在输出中包含任何信息来源。
    5. 必须使用中文输出。
    """

    topic = dspy.InputField(prefix="主题：")
    summary = dspy.InputField(prefix="当前章节简介：")
    focus = dspy.InputField(prefix="聚焦点：")
    information = dspy.InputField(prefix="信息：")
    section_title = dspy.InputField(prefix="章节标题：")
    section = dspy.OutputField(prefix="段落：")


class CitationAdder(dspy.Signature):
    """
    给定句子和信息列表，返回能支持该句子的信息id列表
    """

    sentence = dspy.InputField(prefix="句子：")
    information: List[Information] = dspy.InputField(prefix="信息：")
    cited_indices: List[int] = dspy.OutputField(prefix="引用信息id列表：")


class Summarizer(dspy.Signature):
    """
    用一句话从较广泛的角度概括给定信息
    """

    information = dspy.InputField(prefix="信息：")
    summary = dspy.OutputField(prefix="摘要：")
