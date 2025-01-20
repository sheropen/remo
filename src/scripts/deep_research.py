import dspy
import os
from dotenv import load_dotenv
from pathlib import Path
import argparse
import logging
import concurrent.futures
from config.paths import DEEP_RESEARCH_DIR
from src.agents.retriever import Retriever
from src.agents.planner import Planner
from src.agents.writer import Writer
from src.memory import Memory
from src.utils import setup_logger, timer, Parser
from src.article import Outline, Article
from concurrent.futures import ThreadPoolExecutor

logger = setup_logger()
parent_dir = Path(__file__).parent.parent.parent
load_dotenv(parent_dir / '.env')


MAX_QUERY = 3
MAX_SUBTOPIC = 3
MAX_RESULTS = 20
MAX_SUBTOPIC_EXPLORER_DEPTH = 1
MAX_OUTLINE_DEPTH = 2
MIN_MEMORY_UNITS_FOR_SUBSECTION = 10

model = "openai/deepseek-chat"
better_model = "openai/deepseek-chat"
lm = dspy.LM(model=model, api_key=os.getenv("DEEPSEEK_API_KEY"), api_base=os.getenv("DEEPSEEK_BASE_URL"))
better_lm = dspy.LM(model=better_model, api_key=os.getenv("DEEPSEEK_API_KEY"), api_base=os.getenv("DEEPSEEK_BASE_URL"))

dspy.configure(lm=lm)
dspy.litellm._logging._disable_debugging()
dspy.disable_logging()

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

planner = Planner(engine=better_lm)
retriever = Retriever(engine=lm)
writer = Writer(engine=better_lm)

@timer
def research_topic(topic, summary):
    queries = planner.generate_queries(topic=topic, summary=summary)
    serp_results = retriever.search(queries=queries[:MAX_QUERY], max_results=MAX_RESULTS)
    webpages = retriever.fetch_webpages(webpages=serp_results, timeout=20)
    memory_units = retriever.extract_memory_units(topic=topic, webpages=webpages)
    logger.info(f"Topic: {topic} Generated {len(queries)} queries, {len(serp_results)} search results, {len(webpages)} webpages, {len(memory_units)} memory units")    
    return memory_units


@timer
def recursive_research_subtopic(topic, summary, layer=1):
    note_dict = {}
    memory_units = research_topic(topic=topic, summary=summary)
    summary = writer.summarize_information(information=[unit.content for unit in memory_units])
    note_dict[topic] = summary
    if layer <= MAX_SUBTOPIC_EXPLORER_DEPTH:
        # subtopics take the format of "topic//subtopic"
        subtopics = planner.generate_subtopics(topic=topic, summary=summary)[:MAX_SUBTOPIC]
        subtopics = [f"{topic}//{subtopic}" for subtopic in subtopics]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(recursive_research_subtopic, subtopic, summary, layer=layer+1) for subtopic in subtopics]
            for future in futures:
                _memory_units, _note_dict = future.result()
                memory_units.extend(_memory_units)
                note_dict.update(_note_dict)
    return memory_units, note_dict

@timer
def recursive_generate_outline(memory: Memory, title: str, summary: str, layer: int = 1):
    constraint = {"label": title} if layer != 1 else None
    outline = Outline(title=title, layer=layer)
    
    should_label = True # do not label first
    
    if layer > MAX_OUTLINE_DEPTH or (len(memory.retrieve_information(query=title, k=100, constraint=constraint)) <= MIN_MEMORY_UNITS_FOR_SUBSECTION and should_label):
        return outline
    
    
    if should_label:
        memory_clusters = memory.group_information(constraint=constraint)
    else:
        memory_clusters = memory.group_information(query=title)
        
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(writer.summarize_information, cluster) for cluster in memory_clusters]
        summaries.extend([future.result() for future in futures])
    subsection_titles = planner.generate_outline(topic=title, information=summaries, summary=summary)
    subsection_titles = [f"{title}//{section_title}" for section_title in subsection_titles]
    logger.info(f"Generated {len(subsection_titles)} subsection titles for {title}")
    
    if should_label:
        valid_titles = memory.label_information(labels=subsection_titles, constraint=constraint)
    else:
        valid_titles = subsection_titles
    with ThreadPoolExecutor() as executor:
        futures = []
        for subsection_title in valid_titles:
            future = executor.submit(recursive_generate_outline, memory=memory, summary=summary, title=subsection_title, layer=layer+1)
            futures.append(future)
        for future in futures:
            _outline = future.result()
            outline.insert_child(_outline)
    return outline
    

@timer
def bootstrap(topic: str):
    serp_results = retriever.search(queries=[topic], max_results=20)
    webpages = retriever.fetch_webpages(serp_results, timeout=10)
    memory_units = retriever.extract_memory_units(topic=topic, webpages=webpages)
    information = [unit.content for unit in memory_units]
    summary = writer.summarize_information(information=information)
    return summary

@timer
def main(prompt, skip_research=False, skip_outline=False, skip_write=False, force_recreate=False):
    topic = planner.convert_prompt_to_topic(prompt=prompt)
    logger.info(f"Article Topic: {topic}")
    
    summary = bootstrap(topic=topic)
    logger.info(f"Summary: {summary}")
    
    # memory construction
    memory = Memory(topic=topic, engine=lm, force_recreate=force_recreate)
    if not skip_research:
        memory_units, note_dict = recursive_research_subtopic(topic=topic, summary=summary)
        memory.insert_information(memory_units=memory_units)
        memory.deduplicate_information()
    
    # memory organization
    if not skip_outline:
        outline = recursive_generate_outline(memory=memory, summary=summary, title=topic)
        logger.info(f"Outline: {outline.to_markdown()}")
        
        if True:
            markdown_response = planner.refine_outline(outline=outline)
            logger.info(f"Refined Outline by planner: {markdown_response}")
            refined_outline = Outline.from_markdown(title=topic, markdown=markdown_response)
            logger.info(f"Refined Outline: {refined_outline.to_markdown(show_title=True)}")
            flat_list = refined_outline.to_flatten_list(only_leaf=True)
            logger.info(f"Flat List: {flat_list}")
            memory.label_information(labels=flat_list)
        else:
            refined_outline = outline
    
    # article generation
    article = Article(title=topic)
    
    def write_section(_outline: Outline, summary: str, layer: int):
        section = Article(title=_outline.title, layer=layer)
        # only write the leaf node
        if _outline.children:
            with ThreadPoolExecutor() as executor:
                futures = []
                for subsection in _outline.children:
                    future = executor.submit(write_section, _outline=subsection, summary=summary, layer=layer+1)
                    futures.append((subsection, future))
                for _, future in futures:
                    subsection = future.result()
                    if subsection.subsection_list or subsection.paragraph_list: # if subsection is not empty
                        section.add_subsection(subsection)
                if len(section.subsection_list) == 1: # if only one subsection, then it is the main section
                    subsection = section.subsection_list[0]
                    subsection.title = section.title
                    section = subsection
        else:
            working_context = memory.retrieve_information(query=_outline.title, k=100, constraint={"label": _outline.title})
            logger.info(f"Retrieved {len(working_context)} information for {_outline.title}")
            if len(working_context) > 0:
                information = [unit.content for unit in working_context]
                section_content = writer.write_section(topic=topic, summary=summary, focus="关注主题，忽略与主题无关的信息", information=information, section_title=_outline.title)
                section = Article.from_text(title=_outline.title, layer=layer, text=section_content, working_context=working_context)
                logger.info(f"Section Content for {_outline.title}: {section_content}")
        return section
    
    if not skip_write:  
        article = write_section(_outline=refined_outline, summary=summary, layer=1)
        article.update_reference_dict()
        article.save_to_files(DEEP_RESEARCH_DIR)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep research on a topic')
    parser.add_argument('prompt', type=str, help='Research prompt')
    parser.add_argument('--skip-research', action='store_true', help='Skip research phase')
    parser.add_argument('--skip-outline', action='store_true', help='Skip outline generation')
    parser.add_argument('--skip-write', action='store_true', help='Skip write phase')
    parser.add_argument('--force-recreate', action='store_true', help='Force recreate memory')
    args = parser.parse_args()
    main(args.prompt, skip_research=args.skip_research, skip_outline=args.skip_outline, skip_write=args.skip_write, force_recreate=args.force_recreate)