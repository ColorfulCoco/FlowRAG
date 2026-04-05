# -*- coding: utf-8 -*-
"""
FlowRAG - 基于流程逻辑感知的检索增强生成系统

结合知识图谱推理与向量检索，面向工业SOP文档的问答系统。

Author: CongCongTian
"""

__version__ = "2.0.0"

from src.pipeline.build_pipeline import BuildPipeline
from src.pipeline.retrieval_pipeline import RetrievalPipeline

__all__ = ["BuildPipeline", "RetrievalPipeline"]
