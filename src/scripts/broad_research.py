import dspy
import os
from dotenv import load_dotenv
from pathlib import Path
import asyncio

from src.agents.retriever import Retriever
from src.agents.planner import Planner
from src.utils import setup_logger, timer
from src.data_structure import Information
from config.paths import BROAD_RESEARCH_DIR

logger = setup_logger()

parent_dir = Path(__file__).parent.parent.parent
load_dotenv(parent_dir / '.env')

@timer
def main(prompt):
    max_results = 50
    max_topics = 1
    max_query = 10
    timeout = 20
    batch_size = 100
    
    model = "openai/qwen-plus"
    better_model = "openai/qwen-max"
    lm = dspy.LM(model=model, api_key=os.getenv("ALIYUN_API_KEY"), api_base=os.getenv("ALIYUN_BASE_URL"))
    better_lm = dspy.LM(model=better_model, api_key=os.getenv("ALIYUN_API_KEY"), api_base=os.getenv("ALIYUN_BASE_URL"))
    
    dspy.configure(lm=lm)
    dspy.litellm._logging._disable_debugging()
    dspy.disable_logging()
    
    planner = Planner(engine=better_lm)
    retriever = Retriever(engine=lm)
    
    topic, focus = planner.convert_prompt_with_focus(prompt)
    logger.info(f"Topic: {topic}, Focus: {focus}")
    
    topic_list = [topic]
    topic_list.extend(planner.generate_related_topics(topic=topic))
    logger.info(topic_list)
    
    queries_list = []
    for _topic in topic_list[:max_topics]:
        queries = planner.generate_queries(topic=_topic)
        queries_list.append(queries)
    
    query_set = set()
    total_queries = sum([len(queries) for queries in queries_list])

    current_list = 0
    current_item = 0 
    queries_added = 0

    # Collect queries by cycling through query lists
    while len(query_set) < max_query and queries_added < total_queries:
        # Add query if available at current position
        if current_item < len(queries_list[current_list]):
            query = queries_list[current_list][current_item]
            query_set.add(query)
            queries_added += 1
            
        current_list += 1
        if current_list >= len(queries_list):
            current_list = 0
            current_item += 1
            
    logger.info(query_set)
    
    serp_results = retriever.search(queries=list(query_set), max_results=max_results)
    logger.info(f"Search results: {len(serp_results)}")
    
    # Deduplicate webpages by URL
    unique_webpages = {}
    original_count = len(serp_results)
    for serp_result in serp_results:
        if serp_result.url not in unique_webpages:
            unique_webpages[serp_result.url] = serp_result
    serp_results = list(unique_webpages.values())
    dedup_count = len(serp_results)
    logger.info(f"Deduplication removed {original_count - dedup_count} duplicate webpages ({(original_count - dedup_count)/original_count*100:.1f}% reduction)")
    
    webpages = retriever.fetch_webpages(webpages=serp_results, timeout=timeout)
    logger.info(f"Fetched {len(webpages)} webpages")
    
    memory_units = retriever.extract_memory_units_with_focus(topic=topic, focus=focus, webpages=webpages)
    
    
    information_dict = {}
    important_entities = set()
    for memory_unit in memory_units:
        if memory_unit.content in information_dict:
            information_dict[memory_unit.content].append(memory_unit)
            important_entities.add(memory_unit.content)
        else:
            information_dict[memory_unit.content] = [memory_unit]
    logger.info(f"Deduplication removed {len(memory_units) - len(information_dict)} duplicate information ({(len(memory_units) - len(information_dict))/len(memory_units)*100:.1f}% reduction)")
    
    filtered_entities = set()
    items = list(information_dict.items())
    
    async def process_batch(batch):
        batch_information = [Information(entity=entity, context="；".join([mu.context for mu in mu_list])) for entity, mu_list in batch]
        _entities = planner.filter_information(
            topic=topic,
            focus=focus, 
            information=batch_information
        )
        logger.info(f"Filtered {len(_entities)} entities")
        return _entities

    async def process_all_batches():
        tasks = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            tasks.append(process_batch(batch))
            
        results = await asyncio.gather(*tasks)
        for _entities in results:
            filtered_entities.update(_entities)
            
    asyncio.run(process_all_batches())

    filtered_information_dict = {}
    for entity in filtered_entities:
        try:
            filtered_information_dict[entity] = information_dict[entity]
        except KeyError:
            logger.warning(f"Entity {entity} not found in information_dict")
    
    with open(BROAD_RESEARCH_DIR / f"{topic}.txt", "w") as f:
        for information_content, information in filtered_information_dict.items():
            sources = [information_unit.source for information_unit in information]
            contexts = [information_unit.context.replace("\n", ";") for information_unit in information]
            sources_str = ",".join([f"{source.url}" for source in sources])
            contexts_str = "；".join([f"{context}" for context in contexts])
            f.write(f"{information_content}: {contexts_str} ({sources_str})\n")
            
    os.makedirs(BROAD_RESEARCH_DIR / f"important", exist_ok=True)
    with open(BROAD_RESEARCH_DIR / f"important" / f"{topic}.txt", "w") as f:
        for entity in important_entities:
            sources = [information_unit.source for information_unit in information_dict[entity]]
            contexts = [information_unit.context.replace("\n", ";") for information_unit in information_dict[entity]]
            sources_str = ",".join([f"{source.url}" for source in sources])
            contexts_str = ",".join([f"{context}" for context in contexts])
            f.write(f"{entity}: {contexts_str} ({sources_str})\n")
    
    
    
    
    
if __name__ == "__main__":
    
    # "文化与精神", "产品与技术", "关键事件与里程碑", "公司组织与管理", "行业合作与生态建设", "可持续发展", "消费电子产品与技术", "通信设备与技术", "企业产品与解决方案", "智能汽车解决方案", "数字能源解决方案", "公司组织与机构", "公司经营与管理"
    
    # tasks = ["关键事件与里程碑", "公司组织与管理", "行业合作与生态建设", "可持续发展", "消费电子产品与技术", "通信设备与技术", "企业产品与解决方案", "智能汽车解决方案", "数字能源解决方案", "公司组织与机构", "公司经营与管理"]
    # for task in tasks:
    #     prompt = f"帮我收集华为{task}相关词条，只要词条，不要解释，希望是华为特有的，很特殊的，这些词条将作为百科知识库的主题。记住，不要太泛化，比如这些词条也适用于其他公司的话就不要。同时，不能是太普适性的词，要足够特殊的。"
    #     main(prompt=prompt)
    
    # prompt = "收集品牌或up主征集用户素材制成的视频，视频内容是用户提交的关于某个主题的素材，然后把这些素材串起来升华到一个主题上，我需要视频链接以及关于此类短片的简短描述。所抽取的实体是视频或者征集活动的名字"
    
    prompt = "帮我收集破·地狱的线上看资源，我要链接"
    main(prompt=prompt)
