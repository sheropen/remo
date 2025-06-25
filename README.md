# 华为知识库

## 环境设置

### 前置要求
- Python 3.13.1
- Conda (Anaconda 或 Miniconda)

### 创建Conda环境

```bash
# 创建一个新的conda环境，使用Python 3.13.1
conda create -n huawei-wiki python=3.13.1

# 激活环境
conda activate huawei-wiki
```

### 安装依赖

```bash
# 从requirements.txt安装包
pip install -r requirements.txt
```

### 环境配置

将 `env.sample` 文件中的值替换为你自己的值，并保存为 `.env`。

## 参数配置

系统的关键参数在 `config/constants.py` 中定义，可根据需要进行调整：

### 模型配置
- `REASONING_MODEL`: `"openai/deepseek-r1"` - 推理模型，用于复杂逻辑推理
- `WRITING_MODEL`: `"openai/qwen-max"` - 写作模型，用于文章生成和润色
- `MODEL`: `"openai/deepseek-v3"` - 主模型，用于一般任务

### 搜索和处理限制
- `MAX_WEBPAGE`: `3` - 每次搜索的最大网页数量
- `MAX_QUERY`: `2` - 每个主题的最大查询数量
- `MAX_SUBTOPIC`: `3` - 每个主题的最大子主题数量
- `MAX_SEARCH_DEPTH`: `2` - 搜索的最大深度层级
- `MAX_SUBTOPIC_EXPLORER_DEPTH`: `1` - 子主题探索的最大深度
- `MAX_OUTLINE_DEPTH`: `2` - 大纲生成的最大深度
- `MIN_MEMORY_UNITS_FOR_SUBSECTION`: `10` - 生成子节所需的最小内存单元数
- `MAX_RESULTS`: `10` - 搜索结果的最大数量

### 嵌入模型选项
- `SENTENCE_TRANSFORMER`: 基础句子变换器模型
- `CONAN`: `"TencentBAC/Conan-embedding-v1"` - 腾讯的中文嵌入模型
- `SONAR`: SONAR嵌入模型

**注意**: 调整这些参数时需要考虑API成本和处理时间的平衡。更大的值会获得更全面的结果，但也会增加成本和处理时间。

## 使用方法

### 单主题研究 (`deep_research.py`)

研究单个主题：

```bash
python src/scripts/deep_research.py "<主题>"
```

**示例：**
```bash
python src/scripts/deep_research.py "华为WATCH Ultimate 系列"
```

**可选参数：**
- `--skip-research`: 跳过研究阶段
- `--skip-outline`: 跳过大纲生成阶段
- `--skip-write`: 跳过写作阶段
- `--force-recreate`: 强制重新创建内存数据库

**带参数的示例：**
```bash
python src/scripts/deep_research.py "华为WATCH Ultimate 系列" --skip-research --skip-write
```

### 批量主题研究 (`batch_deep_research.py`)

从文件中处理多个主题：

```bash
python src/scripts/batch_deep_research.py <主题文件>
```

**示例：**
```bash
python src/scripts/batch_deep_research.py topics.txt
```

**主题文件格式：**
创建一个文本文件，每行一个主题：
```
华为WATCH Ultimate
华为Mate 60 Pro
华为P60
```

**可选参数：**
- `--skip-research`: 跳过所有主题的研究阶段
- `--skip-outline`: 跳过所有主题的大纲生成阶段
- `--skip-write`: 跳过所有主题的写作阶段
- `--force-recreate`: 强制重新创建所有主题的内存数据库

**带参数的示例：**
```bash
python src/scripts/batch_deep_research.py topics.txt --skip-outline
```

### 输出结果

研究结果将保存在 `data/deep_research/` 目录中