# -*- coding: utf-8 -*-
"""
FlowRAG 全局配置

LLM / Embedding 配置按阶段划分：
  Build    - 图谱构建、文档扩展、数据集生成
  Retrieve - 向量检索、图谱推理、答案生成
  Evaluate - RAGAS 评估、自定义评估器

各阶段可独立指定 API Key / Base URL / Model，默认继承全局配置。
凭证从环境变量或 .env 文件加载，模板见 .env.example。

Author: CongCongTian
"""

import os
from pathlib import Path

# 加载 .env（需在 os.getenv 之前）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# ============================================================
# 工具函数
# ============================================================

def _env(key: str, default: str = "") -> str:
    """读取环境变量，不存在则返回默认值"""
    return os.getenv(key, default)

# ============================================================
# 全局 API 默认值
# ============================================================
API_KEY         = _env("API_KEY")
BASE_URL        = _env("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME      = _env("MODEL_NAME", "qwen-plus")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "text-embedding-v4")

# ============================================================
# Build 阶段
# ============================================================
BUILD_LLM_API_KEY  = _env("BUILD_LLM_API_KEY", API_KEY)
BUILD_LLM_BASE_URL = _env("BUILD_LLM_BASE_URL", BASE_URL)
BUILD_LLM_MODEL    = _env("BUILD_LLM_MODEL", MODEL_NAME)
BUILD_EMB_API_KEY  = _env("BUILD_EMB_API_KEY", API_KEY)
BUILD_EMB_BASE_URL = _env("BUILD_EMB_BASE_URL", BASE_URL)
BUILD_EMB_MODEL    = _env("BUILD_EMB_MODEL", EMBEDDING_MODEL)

# ============================================================
# Retrieve 阶段
# ============================================================
RETRIEVE_LLM_API_KEY  = _env("RETRIEVE_LLM_API_KEY", BUILD_LLM_API_KEY)
RETRIEVE_LLM_BASE_URL = _env("RETRIEVE_LLM_BASE_URL", BUILD_LLM_BASE_URL)
RETRIEVE_LLM_MODEL    = _env("RETRIEVE_LLM_MODEL", BUILD_LLM_MODEL)
RETRIEVE_EMB_API_KEY  = _env("RETRIEVE_EMB_API_KEY", API_KEY)
RETRIEVE_EMB_BASE_URL = _env("RETRIEVE_EMB_BASE_URL", BASE_URL)
RETRIEVE_EMB_MODEL    = _env("RETRIEVE_EMB_MODEL", EMBEDDING_MODEL)

# ============================================================
# Evaluate 阶段
# ============================================================
EVALUATE_LLM_API_KEY  = _env("EVALUATE_LLM_API_KEY", API_KEY)
EVALUATE_LLM_BASE_URL = _env("EVALUATE_LLM_BASE_URL", BASE_URL)
EVALUATE_LLM_MODEL    = _env("EVALUATE_LLM_MODEL", "qwen-max")
EVALUATE_EMB_API_KEY  = _env("EVALUATE_EMB_API_KEY", API_KEY)
EVALUATE_EMB_BASE_URL = _env("EVALUATE_EMB_BASE_URL", BASE_URL)
EVALUATE_EMB_MODEL    = _env("EVALUATE_EMB_MODEL", "text-embedding-v4")

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIRECTORY = DATA_DIR / "flowrag_store"
LOGIC_TEXT_FOLDER = DATA_DIR / "folder" / "test"
DOCUMENTS_FOLDER = DATA_DIR / "synthetic_dataset" / "sop_documents"
MERMAID_OUTPUT_DIR = PERSIST_DIRECTORY / "mermaid_visualization"

# ============================================================
# 向量库
# ============================================================
VECTOR_BACKEND = "faiss"  # faiss 或 chroma

# ============================================================
# 文本分块（优先级：段落 > 句子 > 语义 > 基础）
# ============================================================
SPLIT_BY_PARAGRAPH = True
SPLIT_BY_SENTENCE = False
USE_SEMANTIC_SPLIT = False
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MAX_SENTENCES_PER_CHUNK = 2

# ============================================================
# 知识挂载
# ============================================================
MAX_ENRICHMENT_CHUNKS = 6

# ============================================================
# 检索参数
# ============================================================
FUSION_STRATEGY = "rrf"          # rrf / weighted / cascade
GRAPH_WEIGHT = 0.6
EXPAND_HOPS = 1
INCLUDE_CAUSAL_CHAIN = True
MAX_REASONING_ITERATIONS = 4
TOP_ANCHOR_NODES = 2
TOP_K = 5

# ============================================================
# 上下文 token 预算
# ============================================================
CONTEXT_TOKEN_LIMIT = 7000

# ============================================================
# API 重试（应对 429 限流、网络抖动等）
# ============================================================
API_RETRY_INTERVAL_SECONDS = int(_env("API_RETRY_INTERVAL_SECONDS", "10"))
API_RETRY_MAX_ATTEMPTS = int(_env("API_RETRY_MAX_ATTEMPTS", "10"))

# ============================================================
# 日志级别：False=仅关键日志+进度条，True=全量详细日志
# ============================================================
LOG_VERBOSE = _env("LOG_VERBOSE", "0").strip() in ("1", "true", "True", "yes")

# ============================================================
# 参数校验
# ============================================================
assert CHUNK_SIZE > 0, f"CHUNK_SIZE must be > 0, got {CHUNK_SIZE}"
assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE, f"CHUNK_OVERLAP must be in [0, CHUNK_SIZE), got {CHUNK_OVERLAP}"
assert MAX_ENRICHMENT_CHUNKS > 0, f"MAX_ENRICHMENT_CHUNKS must be > 0, got {MAX_ENRICHMENT_CHUNKS}"
assert CONTEXT_TOKEN_LIMIT > 0, f"CONTEXT_TOKEN_LIMIT must be > 0, got {CONTEXT_TOKEN_LIMIT}"
assert API_RETRY_INTERVAL_SECONDS >= 0, f"API_RETRY_INTERVAL_SECONDS must be >= 0, got {API_RETRY_INTERVAL_SECONDS}"
assert API_RETRY_MAX_ATTEMPTS >= 0, f"API_RETRY_MAX_ATTEMPTS must be >= 0, got {API_RETRY_MAX_ATTEMPTS}"

# ============================================================
# Mermaid 解析
# ============================================================
USE_LLM_MERMAID_PARSE = False
