# -*- coding: utf-8 -*-
"""
检索模块

提供检索缓存管理，支持断点续传和结果复用。

Author: CongCongTian
"""

from .retrieval_cache import RetrievalCache, RetrievalRecord

__all__ = ["RetrievalCache", "RetrievalRecord"]
