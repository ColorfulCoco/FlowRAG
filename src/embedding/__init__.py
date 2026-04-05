# -*- coding: utf-8 -*-
"""
嵌入模块

提供文本嵌入和结构上下文感知嵌入能力。

Author: CongCongTian
"""

from .embedder import Embedder
from .context_embedder import ContextEmbedder

__all__ = ["Embedder", "ContextEmbedder"]
