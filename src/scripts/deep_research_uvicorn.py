from fastapi import FastAPI, HTTPException
from pathlib import Path
import os
from config.paths import DEEP_RESEARCH_DIR
from src.utils import Parser
import asyncio
from typing import Dict
import time

app = FastAPI()

# 用于追踪正在处理的任务
running_tasks: Dict[str, float] = {}
RUNNING_FILE = Path("data/running.txt")
TIMEOUT = 3600  # 1小时超时


def is_article_exists(topic: str) -> bool:
    """检查文章是否已经生成"""
    safe_title = Parser.safe_title(topic)
    rewritten_path = DEEP_RESEARCH_DIR / "rewritten" / f"{safe_title}.txt"
    return rewritten_path.exists()


def is_topic_running(topic: str) -> bool:
    """检查主题是否正在处理中"""
    # 清理超时的任务
    current_time = time.time()
    expired_topics = [
        t
        for t, start_time in running_tasks.items()
        if current_time - start_time > TIMEOUT
    ]
    for t in expired_topics:
        del running_tasks[t]

    # 检查是否在运行中
    if topic in running_tasks:
        return True

    # 检查running.txt文件
    if RUNNING_FILE.exists():
        with open(RUNNING_FILE, "r", encoding="utf-8") as f:
            running_topics = f.read().splitlines()
            return topic in running_topics

    return False


def add_topic_to_running(topic: str):
    """添加主题到运行列表"""
    running_tasks[topic] = time.time()

    # 确保目录存在
    RUNNING_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 添加到running.txt
    with open(RUNNING_FILE, "a+", encoding="utf-8") as f:
        f.seek(0)
        topics = f.read().splitlines()
        if topic not in topics:
            f.write(f"{topic}\n")


@app.post("/research/{topic}")
async def research_topic(topic: str):
    # 检查文章是否已存在
    if is_article_exists(topic):
        safe_title = Parser.safe_title(topic)
        with open(
            DEEP_RESEARCH_DIR / "rewritten" / f"{safe_title}.txt", "r", encoding="utf-8"
        ) as f:
            content = f.read()
        return {
            "status": "completed",
            "message": "文章已经生成，请点击Display Article按钮查看",
        }

    # 检查是否正在处理中
    if is_topic_running(topic):
        return {"status": "processing", "message": "文章正在生成中，请稍后再试"}

    # 添加到运行列表
    add_topic_to_running(topic)

    # 启动异步任务处理
    try:
        # 这里可以使用asyncio.create_subprocess_exec来异步执行deep_research.py
        cmd = ["python", "src/scripts/deep_research.py", topic]
        process = await asyncio.create_subprocess_exec(*cmd)
        return {"status": "started", "message": "已开始生成文章"}
    except Exception as e:
        # 如果启动失败，从运行列表中移除
        if topic in running_tasks:
            del running_tasks[topic]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
