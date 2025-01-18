import dspy
from typing import List
from src.utils import timer
from src.data_structure import MemoryUnit


class Writer:
    def __init__(self, engine: dspy.dsp.LM):
        self.engine = engine

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
    def write_section(self, topic: str, summary: str, focus: str, information: List[str], section_title: str):
        f = dspy.Predict(SectionWriter)
        with dspy.settings.context(lm=self.engine):
            response = f(topic=topic, summary=summary, focus=focus, information=information, section_title=section_title)
        return response.section
    
    @timer
    def summarize_information(self, information: List[str]):
        f = dspy.Predict(Summarizer)
        with dspy.settings.context(lm=self.engine):
            response = f(information=information)
        return response.summary

class ArticleWriter(dspy.Signature):
    """
    给定主题、聚焦点、信息，以维基百科风格输出文章
    生成规则
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
    4. 必须使用中文输出。
    """
    topic = dspy.InputField(prefix="主题：")
    summary = dspy.InputField(prefix="当前章节简介：")
    focus = dspy.InputField(prefix="聚焦点：")
    information = dspy.InputField(prefix="信息：")
    section_title = dspy.InputField(prefix="章节标题：")
    section = dspy.OutputField(prefix="段落：")
    
class Summarizer(dspy.Signature):
    """
    用一句话从较广泛的角度概括给定信息
    """
    information = dspy.InputField(prefix="信息：")
    summary = dspy.OutputField(prefix="摘要：")