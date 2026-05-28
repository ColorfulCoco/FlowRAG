# -*- coding: utf-8 -*-
"""
向量存储模块

封装 FAISS / ChromaDB 后端，提供统一的向量检索接口。

Author: CongCongTian
"""

from .vector_db import VectorDB

__all__ = ["VectorDB"]
