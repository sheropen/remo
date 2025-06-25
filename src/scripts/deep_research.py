import dspy
import os
import json
from dotenv import load_dotenv
from pathlib import Path
import argparse
import logging
import concurrent.futures
from config.paths import DEEP_RESEARCH_DIR
from config.constants import Constants
from src.agents.retriever import Retriever
from src.agents.planner import Planner
from src.agents.writer import Writer
from src.memory import Memory
from src.utils import setup_logger, timer, Parser
from src.article import Outline, Article
from concurrent.futures import ThreadPoolExecutor

logger = setup_logger()
parent_dir = Path(__file__).parent.parent.parent
load_dotenv(parent_dir / ".env")

lm = dspy.LM(
    model=Constants.MODEL,
    api_key=os.getenv("ALIYUN_API_KEY"),
    api_base=os.getenv("ALIYUN_BASE_URL"),
)
reasoning_lm = dspy.LM(
    model=Constants.REASONING_MODEL,
    api_key=os.getenv("ALIYUN_API_KEY"),
    api_base=os.getenv("ALIYUN_BASE_URL"),
)
writing_lm = dspy.LM(
    model=Constants.WRITING_MODEL,
    api_key=os.getenv("ALIYUN_API_KEY"),
    api_base=os.getenv("ALIYUN_BASE_URL"),
)


dspy.configure(lm=lm)
dspy.litellm._logging._disable_debugging()
dspy.disable_logging()

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

planner = Planner(engine=lm, reasoning_engine=reasoning_lm)
retriever = Retriever(engine=lm)
writer = Writer(engine=lm, better_engine=writing_lm)


@timer
def research_topic(topic, summary):
    queries = planner.generate_queries(topic=topic, summary=summary)
    serp_results = retriever.search(queries=queries[:Constants.MAX_QUERY], max_results=Constants.MAX_RESULTS)
    webpages = retriever.fetch_webpages(webpages=serp_results, timeout=10)
    memory_units = retriever.extract_memory_units(topic=topic, webpages=webpages)
    logger.info(
        f"Topic: {topic}, Generated {Constants.MAX_QUERY}/{len(queries)} queries, {len(serp_results)} search results, {len(webpages)} webpages, {len(memory_units)} memory units"
    )
    return memory_units


@timer
def recursive_research_subtopic(topic, summary, layer=1):
    note_dict = {}
    memory_units = research_topic(topic=topic, summary=summary)
    summary = writer.summarize_information(
        information=[unit.content for unit in memory_units]
    )
    note_dict[topic] = summary
    if layer <= Constants.MAX_SUBTOPIC_EXPLORER_DEPTH:
        # subtopics take the format of "topic//subtopic"
        subtopics = planner.generate_subtopics(topic=topic, summary=summary)[
            :Constants.MAX_SUBTOPIC
        ]
        subtopics = [f"{topic}//{subtopic}" for subtopic in subtopics]
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    recursive_research_subtopic, subtopic, summary, layer=layer + 1
                )
                for subtopic in subtopics
            ]
            for future in futures:
                _memory_units, _note_dict = future.result()
                memory_units.extend(_memory_units)
                note_dict.update(_note_dict)
    return memory_units, note_dict


@timer
def recursive_generate_outline(
    memory: Memory, title: str, summary: str, layer: int = 1
):
    constraint = {"label": title} if layer != 1 else None
    outline = Outline(title=title, layer=layer)

    should_label = True  # do not label first

    if layer > Constants.MAX_OUTLINE_DEPTH or (
        len(memory.retrieve_information(query=title, k=100, constraint=constraint))
        <= Constants.MIN_MEMORY_UNITS_FOR_SUBSECTION
        and should_label
    ):
        return outline

    if should_label:
        memory_clusters = memory.group_information(constraint=constraint)
    else:
        memory_clusters = memory.group_information(query=title, n_clusters=5)

    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(writer.summarize_information, cluster)
            for cluster in memory_clusters
        ]
        summaries.extend([future.result() for future in futures])
    subsection_titles = planner.generate_outline(
        topic=title, information=summaries, summary=summary
    )
    subsection_titles = [
        f"{title}//{section_title}" for section_title in subsection_titles
    ]
    logger.info(f"Generated {len(subsection_titles)} subsection titles for {title}")

    if should_label:
        valid_titles = memory.label_information(
            labels=subsection_titles, constraint=constraint
        )
    else:
        valid_titles = subsection_titles
    with ThreadPoolExecutor() as executor:
        futures = []
        for subsection_title in valid_titles:
            future = executor.submit(
                recursive_generate_outline,
                memory=memory,
                summary=summary,
                title=subsection_title,
                layer=layer + 1,
            )
            futures.append(future)
        for future in futures:
            _outline = future.result()
            outline.insert_child(_outline)
    return outline


@timer
def bootstrap(topic: str):
    serp_results = retriever.search(queries=[topic], max_results=Constants.MAX_RESULTS)
    webpages = retriever.fetch_webpages(serp_results, timeout=10)
    memory_units = retriever.extract_memory_units(topic=topic, webpages=webpages)
    information = [unit.content for unit in memory_units]
    summary = writer.summarize_information(information=information)
    return summary


@timer
def add_citations(article: Article):
    with ThreadPoolExecutor() as executor:
        futures = []
        for paragraph in article.paragraph_list:
            for sentence in paragraph.sentence_list:
                future = executor.submit(
                    writer.find_citations,
                    sentence=sentence,
                    information_list=article.working_context,
                )
                futures.append((sentence, future))

        for sentence, future in futures:
            citation_indices = future.result()
            sentence.citation_list = [
                article.working_context[i] for i in citation_indices
            ]

    with ThreadPoolExecutor() as executor:
        futures = []
        for subsection in article.subsection_list:
            future = executor.submit(add_citations, subsection)
            futures.append(future)
        for future in futures:
            future.result()


@timer
def main(
    topic,
    skip_research=False,
    skip_outline=False,
    skip_write=False,
    force_recreate=False,
):
    logger.info(f"Article Topic: {topic}")

    summary = bootstrap(topic=topic)
    logger.info(f"Summary: {summary}")

    # memory construction
    memory = Memory(topic=topic, engine=lm, force_recreate=force_recreate)
    if not skip_research:
        memory_units, note_dict = recursive_research_subtopic(
            topic=topic, summary=summary
        )
        for k, v in note_dict.items():
            logger.info(f"{k}: {v}")
        memory.insert_information(memory_units=memory_units)
        memory.deduplicate_information()

    # memory organization
    outline_dir = DEEP_RESEARCH_DIR / "outline"
    os.makedirs(outline_dir, exist_ok=True)
    if skip_outline:
        with open(outline_dir / f"{topic}.json", "r", encoding="utf-8") as f:
            refined_outline = Outline.from_dict(data=json.load(f))
    else:
        outline = recursive_generate_outline(
            memory=memory, summary=summary, title=topic
        )
        logger.info(f"Outline: {outline.to_markdown()}")

        refined_outline_markdown = planner.refine_outline(outline=outline)
        refined_outline = Outline.from_markdown(
            title=topic, markdown=refined_outline_markdown
        )
        logger.info(f"Refined Outline: {refined_outline.to_markdown(show_title=True)}")

        rearranged_outline = refined_outline

        with open(
            outline_dir / f"{Parser.safe_title(topic)}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(rearranged_outline.to_dict(), f, ensure_ascii=False)

        flat_list = rearranged_outline.to_flatten_list(only_leaf=True)
        logger.info(f"Flat List: {flat_list}")
        memory.label_information(labels=flat_list)

    def write_section(_outline: Outline, summary: str, layer: int):
        section = Article(title=_outline.title, layer=layer, working_context=[])
        if len(_outline.children) == 0:
            working_context = memory.retrieve_information(
                query=_outline.title, k=100, constraint={"label": _outline.title}
            )
            logger.info(
                f"Retrieved {len(working_context)} information for {_outline.title}"
            )
            if len(working_context) > 0:
                information = [unit.content for unit in working_context]
                section_content = writer.write_section(
                    topic=topic,
                    summary=summary,
                    focus="关注主题，忽略与主题无关的信息",
                    information=information,
                    section_title=_outline.title,
                )
                section = Article.from_text(
                    title=_outline.title,
                    layer=layer,
                    text=section_content,
                    working_context=working_context,
                )
                logger.info(f"Section Content for {_outline.title}: {section_content}")
        else:
            with ThreadPoolExecutor() as executor:
                futures = []
                for subsection in _outline.children:
                    future = executor.submit(
                        write_section,
                        _outline=subsection,
                        summary=summary,
                        layer=layer + 1,
                    )
                    futures.append((subsection, future))
                for subsection, future in futures:
                    result = future.result()
                    if (
                        result.subsection_list or result.paragraph_list
                    ):  # if subsection is not empty
                        section.add_subsection(result)
                if (
                    len(section.subsection_list) == 1
                ):  # if only one subsection, then it is the main section
                    subsection = section.subsection_list[0]
                    subsection.title = section.title
                    subsection.layer = section.layer
                    section = subsection

        return section

    # article generation
    if not skip_write:
        working_context = memory.retrieve_information(query=f"{topic}的简介", k=100)
        abstract = writer.write_section(
            topic=f"{topic}",
            summary=summary,
            focus="关注最重要的信息，写成一小段简短的介绍，特别是主题的背景介绍",
            information=[unit.content for unit in working_context],
            section_title="简介",
        )
        abstract_article = Article.from_text(
            title=f"{topic}",
            layer=1,
            text=abstract,
            working_context=working_context,
        )
        article = write_section(_outline=refined_outline, summary=summary, layer=1)
        article.working_context.extend(abstract_article.working_context)
        article.paragraph_list = abstract_article.paragraph_list
        article.update_reference_dict()
        add_citations(article)

        # Save both original version
        article.save_to_files(DEEP_RESEARCH_DIR)

        # Rewrite the article in a more engaging style
        article_content = article.__repr__(show_citation=False, show_reference=False)
        rewritten_content = writer.rewrite_article(article=article_content)

        os.makedirs(DEEP_RESEARCH_DIR / "rewritten", exist_ok=True)
        with open(
            DEEP_RESEARCH_DIR / "rewritten" / f"{Parser.safe_title(topic)}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(rewritten_content)
            f.write("\n\n## 参考文献\n")
            for doc_id in range(1, article.doc_cnt):
                f.write(f"[{doc_id}] {article.reference_dict[doc_id]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep research on a topic")
    parser.add_argument("topic", type=str, help="Research topic")
    parser.add_argument(
        "--skip-research", action="store_true", help="Skip research phase"
    )
    parser.add_argument(
        "--skip-outline", action="store_true", help="Skip outline generation"
    )
    parser.add_argument("--skip-write", action="store_true", help="Skip write phase")
    parser.add_argument(
        "--force-recreate", action="store_true", help="Force recreate memory"
    )
    args = parser.parse_args()
    main(
        args.topic,
        skip_research=args.skip_research,
        skip_outline=args.skip_outline,
        skip_write=args.skip_write,
        force_recreate=args.force_recreate,
    )
