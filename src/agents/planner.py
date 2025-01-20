import dspy
from typing import List
from src.data_structure import Information
from src.article import Outline
from src.utils import Parser, setup_logger

logger = setup_logger()

class Planner:
    def __init__(self, engine: dspy.dsp.LM):
        self.engine = engine 

    def convert_prompt_with_focus(self, prompt: str):
        f = dspy.Predict(PromptConverterWithFocus)
        with dspy.settings.context(lm=self.engine):
            response = f(prompt=prompt)
        return response.topic, response.focus
    
    def convert_prompt_to_topic(self, prompt: str):
        f = dspy.Predict(PromptConverter)
        with dspy.settings.context(lm=self.engine):
            response = f(prompt=prompt)
        return response.topic
    
    def generate_queries(self, topic: str, summary: str=None):
        f = dspy.Predict(QueryGenerator)
        with dspy.settings.context(lm=self.engine):
            response = f(topic=topic, summary=summary)
        return response.queries
    
    def generate_outline(self, topic: str, information: List[str], summary: str=None):
        f = dspy.Predict(OutlineGenerator)
        with dspy.settings.context(lm=self.engine):
            response = f(topic=topic, information=information, summary=summary)
        return [Parser.clean_section_name(section_name) for section_name in response.outline]
    
    def generate_subtopics(self, topic: str, summary: str):
        f = dspy.Predict(SubtopicGenerator)
        with dspy.settings.context(lm=self.engine):
            response = f(topic=topic, summary=summary)
        return response.subtopics
    
    def generate_related_topics(self, topic: str):
        f = dspy.Predict(RelatedTopicGenerator)
        with dspy.settings.context(lm=self.engine):
            response = f(topic=topic)
        return response.related_topics
    
    def refine_outline(self, outline: Outline):
        f = dspy.ChainOfThought(OutlineRefiner)
        with dspy.settings.context(lm=self.engine):
            response = f(outline=outline)
        logger.info(f"Reasoning: {response.reasoning}")
        return response.refined_outline
    
    # use weaker model to filter information (for broad search, extracting entities)
    def filter_information(self, topic: str, focus: str, information: List[Information]):
        f = dspy.Predict(InformationFilter)
        response = f(topic=topic, focus=focus, information=information)
        return response.entities
    
    def rearrange_sections(self, outline: Outline, summary: str):
        f = dspy.Predict(SectionRearranger)
        response = f(outline=outline, summary=summary)
        return response.rearranged_outline
    
class PromptConverter(dspy.Signature):
    """
    给定用户的原始prompt，提取出主题，以作为符合用户要求的维基百科文章的标题。
    生成规则：
    1. 主题必须与prompt高度相关，避免过于宽泛的表述，应明确包含用户想要了解的具体领域或方面
    2. 用中文输出
    """
    prompt = dspy.InputField(prefix="原始prompt：")
    topic = dspy.OutputField(prefix="主题：")
    
class PromptConverterWithFocus(dspy.Signature):
    """
    给定用户的原始prompt，提取出主题，以作为符合用户要求的百科文章的标题。
    生成规则：
    1. 主题必须与prompt高度相关，避免过于宽泛的表述，应明确包含用户想要了解的具体领域或方面
    2. 聚焦点是用户想要收集的信息类型与详情，用简短的句子描述用户的需求
    3. 用中文输出
    """
    prompt = dspy.InputField(prefix="原始prompt：")
    topic = dspy.OutputField(prefix="主题：")
    focus = dspy.OutputField(prefix="聚焦点：")

class QueryGenerator(dspy.Signature):
    """
    给定主题，生成多样化的谷歌搜索条目
    生成规则：
    1. 每个条目必须是用空格分开的关键词系列
    2. 每个条目的实体必须是主题中的实体（保留名词）
    3. 生成多样化的关键词，关键词不要重复，多用不同的近义词
    """
    topic = dspy.InputField(prefix="主题：")
    summary = dspy.InputField(prefix="当前主题简介：")
    queries: List[str] = dspy.OutputField(prefix="查询词列表：")

class RelatedTopicGenerator(dspy.Signature):
    """
    给定主题，生成相关主题
    生成规则：
    1. 相关主题必须和原主题主体一致，但不要使用同样的关键词
    2. 相关主题的描述要与原主题的描述有所区别，不要使用同样的描述
    3. 用中文输出
    """
    topic = dspy.InputField(prefix="主题：")
    related_topics: List[str] = dspy.OutputField(prefix="相关主题列表：")

class SubtopicGenerator(dspy.Signature):
    """
    给定主题以及当前主题的总结，生成多样化的子主题
    生成规则：
    1. 每个子主题必须与主题高度相关
    2. 每个子主题必须与当前主题的总结有所区别
    3. 用中文输出
    """
    topic = dspy.InputField(prefix="主题：")
    summary = dspy.InputField(prefix="当前主题总结：")
    subtopics: List[str] = dspy.OutputField(prefix="子主题列表：")
    
class OutlineGenerator(dspy.Signature):
    """
    给定当前章节标题和信息，以维基百科风格将当前章节内容细分为多个子章节
    生成规则：
    1. 只生成一级大纲，不进一步细分
    2. 不要生成“概述”作为子章节
    3. 确保子章节组织与顺序符合逻辑，以保证文章的连贯性与可读性
    4. 用中文输出
    """
    topic = dspy.InputField(prefix="当前章节标题：")
    information = dspy.InputField(prefix="信息：")
    summary = dspy.InputField(prefix="当前章节简介：")
    outline: List[str] = dspy.OutputField(prefix="子章节标题列表：")
    
class OutlineRefiner(dspy.Signature):
    """
    以markdown格式（## 章节标题，### 子章节标题）以及维基风格重写给定的大纲，以确保可读性与逻辑流畅性
    生成规则：
    1. 不要生成过多的子章节，同时确保子章节之间没有任何重复
    2. 保证大纲按照人类的阅读习惯，比如逻辑关系以及阅读顺序
    3. 不要生成章节序号
    4. 用中文输出
    """
    outline = dspy.InputField(prefix="大纲：")
    refined_outline = dspy.OutputField(prefix="完善后的大纲：")

class InformationFilter(dspy.Signature):
    """
    给定信息（包含实体和上下文），过滤出符合主题与聚焦点的实体
    """
    topic = dspy.InputField(prefix="主题：")
    focus = dspy.InputField(prefix="聚焦点：")
    information = dspy.InputField(prefix="信息：")
    entities: List[str] = dspy.OutputField(prefix="实体列表：")
    
    
class SectionRearranger(dspy.Signature):
    """
    给定大纲和主题简介，按照人类的阅读顺序重新排列大纲章节顺序，以确保大纲的逻辑性和可读性
    以markdown格式（## 章节标题，### 子章节标题）输出
    """
    outline = dspy.InputField(prefix="大纲：")
    summary = dspy.InputField(prefix="主题简介：")
    rearranged_outline = dspy.OutputField(prefix="重新排列后的大纲：")
