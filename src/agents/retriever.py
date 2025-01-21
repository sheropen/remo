import json
import os
import aiohttp
import asyncio
from typing import List, Union
import dspy
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from config.paths import SERP_DIR, WEBPAGE_DIR
from src.utils import Parser, timer, setup_logger
from src.data_structure import MemoryUnit, Webpage, Information

logger = setup_logger()

class Retriever:
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.extract_information = dspy.Predict(InformationExtractor)
        self.extract_information_with_focus = dspy.Predict(InformationExtractorWithFocus)
        self.engine = engine
        
    @timer
    def search(self, queries: List[str], max_concurrent: int = 5, max_results: int = 100):
        '''
        Performs a search using ValueSERP API and returns a list of 
        search results containing title, url, and snippet.
        '''
        async def fetch_serp(query: str, semaphore: asyncio.Semaphore):
            async with semaphore:  # Rate limit concurrent API calls
                try:
                    output_path = os.path.join(SERP_DIR, f"{Parser.to_hash(query)}.json")
                    if os.path.exists(output_path):
                        with open(output_path, "r", encoding="utf-8") as f:
                            serp_results = json.load(f)
                            logger.info(f"Loaded cached results for query: {query}")
                    else:
                        params = {
                            "api_key": os.getenv("SERP_API_KEY"),
                            "num": max_results,
                            "q": query,
                        }
                        async with aiohttp.ClientSession() as session:
                            async with session.get("https://api.valueserp.com/search", params=params) as response:
                                if response.status != 200:
                                    logger.error(f"API error for query {query}: {response.status}")
                                    return []
                                serp_results = await response.json()
                        
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(serp_results, f, ensure_ascii=False)
                    
                    parsed_results = []
                    for result in serp_results.get("organic_results", []):
                        webpage = Webpage(
                            title=result["title"],
                            content=None,
                            url=result["link"],
                            snippet=result.get("snippet", None),
                            date=result.get("date_utc", None),
                            query=query
                        )
                        parsed_results.append(webpage)
                    logger.info(f"Found {len(parsed_results)} results for query: {query}")
                    return parsed_results
                except Exception as e:
                    logger.error(f"Error processing query {query}: {str(e)}")
                    return []

        async def run_searches():
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [fetch_serp(query, semaphore) for query in queries]
            results = await asyncio.gather(*tasks)
            return results

        # Run async code in synchronous context
        results = asyncio.run(run_searches())
        all_results = [item for sublist in results for item in sublist]
        
        # Deduplicate results based on URL
        seen_urls = set()
        deduplicated_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated_results.append(result)
                
        logger.info(f"Found {len(deduplicated_results)}/{len(all_results)} unique results from {len(queries)} queries after deduplication")
        return deduplicated_results

    @timer
    def fetch_webpages(self, webpages: List[Webpage], timeout: int = 5):
        """
        Scrapes content from a list of URLs using aiohttp.
        Returns a dictionary mapping URLs to their HTML content.
        """
        async def fetch_url(session, webpage):
            
            try:
                async with session.get(webpage.url, timeout=timeout) as response:
                    if response.status == 200:
                        response_text = await response.text(encoding=response.get_encoding())
                        soup = BeautifulSoup(response_text, 'html.parser')
                        
                        for script in soup(["script", "style"]):
                            script.decompose()
                            
                        text = soup.get_text(separator='\n', strip=True)
                        
                        lines = (line.strip() for line in text.splitlines())
                        text = '\n'.join(line for line in lines if line)
                        
                        webpage.content = text
                        return webpage
                    return None
            except Exception as e:
                print(f"Error fetching {webpage.url}: {e}")
                return None

        async def fetch_all():
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_url(session, webpage) for webpage in webpages]
                results = await asyncio.gather(*tasks)
                return [webpage for webpage in results if webpage is not None]

        return asyncio.run(fetch_all())

    
    @timer
    def extract_memory_units(self, topic: str, webpages: List[Webpage]):
        
        def extract_units_from_chunk(args):
            webpage, chunk = args
            try:
                response = self.extract_information(
                    title=webpage.title,
                    date=webpage.date,
                    content=chunk,
                    topic=topic,
                    link=webpage.url
                )
                return response.information
            except Exception as e:
                logger.error(f"Error processing chunk from {webpage.url}: {str(e)}")
                return None

        def extract_units_from_webpage(webpage: Webpage):
            chunks = Parser.chunk_text(webpage.content, 20000)
            chunk_args = [(webpage, chunk) for chunk in chunks]
            
            with ThreadPoolExecutor() as executor:
                chunk_results = list(executor.map(extract_units_from_chunk, chunk_args))
            
            memory_units = []
            for chunk_result in chunk_results:
                if chunk_result is not None:
                    for unit in chunk_result:
                        memory_units.append(MemoryUnit(
                                content=unit,
                                context=None,
                                topic=topic,
                                source=webpage
                            ))
            return memory_units

        with ThreadPoolExecutor() as executor:
            webpage_results = list(executor.map(extract_units_from_webpage, webpages))
        
        memory_units = []
        for webpage_units in webpage_results:
            if webpage_units is not None:
                memory_units.extend(webpage_units)
        return memory_units
    
    
    @timer
    def extract_memory_units_with_focus(self, topic: str, focus: str, webpages: List[Webpage]):
        results = set()
        
        def extract_information_from_chunk(args):
            webpage, chunk = args
            try:
                response = self.extract_information_with_focus(
                    topic=topic,
                    focus=focus,
                    content=chunk
                )
                return response.information
            except Exception as e:
                logger.error(f"Error processing chunk from {webpage.url}: {str(e)}")
                return None
        
        def extract_information_from_webpage(webpage: Webpage):
            chunks = Parser.chunk_text(webpage.content, 20000)
            chunk_args = [(webpage, chunk) for chunk in chunks]
            
            with ThreadPoolExecutor() as executor:
                chunk_results = list(executor.map(extract_information_from_chunk, chunk_args))
            
            memory_units = []
            for chunk_result in chunk_results:
                if chunk_result is not None:
                    for information_unit in chunk_result:
                        memory_units.append(MemoryUnit(
                            content=information_unit.entity,
                            context=information_unit.context,
                            source=webpage
                        ))
            return memory_units
        with ThreadPoolExecutor() as executor:
            webpage_results = list(executor.map(extract_information_from_webpage, webpages))
            
        memory_units = []
        for webpage_units in webpage_results:
            if webpage_units is not None:
                memory_units.extend(webpage_units)
        return memory_units


class InformationExtractor(dspy.Signature):
    """
    给定一个网页文本，从网页内容中提取与给定主题完全相关的事实。
    提取规则：
    1. 提取所有与主题完全相关的事实
    2. 每条事实是完整的独立的陈述句或段落，其中必须包含：
        - 命名实体（人名、组织、地点、产品等）
        - 事实的时间信息，“今”等描述可由网页创建时间推断
    3. 不要出现任何代词，如“它”、“他们”、“这个”、“那个”、“该”等，应补全相应的命名实体
    4. 如果内容与主题无关，返回空列表 []
    """
    title = dspy.InputField(prefix="网页标题：")
    link = dspy.InputField(prefix="网页链接：")
    date = dspy.InputField(prefix="网页日期：")
    content = dspy.InputField(prefix="网页内容：")
    
    topic = dspy.InputField(prefix="主题：")
    information: List[str] = dspy.OutputField(prefix="提取的信息：")
   
class InformationExtractorWithFocus(dspy.Signature):
    """
    给定主题和聚焦点，从网页内容中提取与主题和聚焦点相关的信息（若有）。
    提取规则：
    1. 只提取所有与主题相关的信息
    2. 必须按照聚焦点的指示提取相关信息
    3. 如果内容与主题和聚焦点无关，直接返回空列表 []
    """
    topic = dspy.InputField(prefix="主题：")
    focus = dspy.InputField(prefix="聚焦点：")
    content = dspy.InputField(prefix="网页内容：")
    information: List[Information] = dspy.OutputField(prefix="提取的信息：")
        
    