# -*- coding: utf-8 -*-
"""
嵌入模型封装

支持 OpenAI API 兼容接口和本地模型（sentence-transformers）。

Author: CongCongTian
"""

import os
import time
import logging
import threading
from typing import List, Union, Optional
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """嵌入模型基类"""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        生成文本嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            嵌入向量数组，shape为 (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        生成查询嵌入向量（某些模型对查询和文档使用不同的嵌入方式）
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量，shape为 (embedding_dim,)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """嵌入向量维度"""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI API兼容的嵌入模型（支持阿里通义千问等）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 10,  # 阿里云通义千问限制每批最多10个
        token_stats=None,
        embedding_type: str = "unknown",
    ):
        """
        Args:
            api_key: API 密钥，优先级：参数 > 环境变量 > config.py
            base_url: API 基础 URL，同上优先级
            model_name: 模型名称
            batch_size: 批处理大小
            token_stats: token 统计对象
            embedding_type: 嵌入类型（text/keyword/node），用于分类统计
        """
        from src.config import API_KEY, BASE_URL, EMBEDDING_MODEL
        default_api_key = API_KEY
        default_base_url = BASE_URL
        default_model = EMBEDDING_MODEL
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or default_api_key
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or default_base_url
        self.model_name = model_name or default_model
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError("API密钥未设置！请在 src/config.py 中设置 API_KEY 或设置环境变量 OPENAI_API_KEY")
        
        from src.utils.openai_client import create_openai_client
        self.client = create_openai_client(api_key=self.api_key, base_url=self.base_url)
        
        # 已知模型维度映射，避免初始化时调 API（可能被限流）
        _KNOWN_DIMENSIONS = {
            "text-embedding-v4": 1024,
            "text-embedding-v3": 1024,
            "text-embedding-v2": 1536,
            "text-embedding-v1": 1536,
        }
        self._dimension = _KNOWN_DIMENSIONS.get(self.model_name, None)
        
        logger.info(f"[Embedder] 初始化完成 | 模型: {self.model_name} | batch_size: {self.batch_size}")
        
        self.token_stats = token_stats
        # threading.local() 避免并发线程间的 embedding_type 竞争
        self._default_embedding_type = embedding_type
        self._thread_local = threading.local()
    
    @property
    def embedding_type(self) -> str:
        """当前线程的 embedding 类型"""
        return getattr(self._thread_local, 'embedding_type', self._default_embedding_type)
    
    @embedding_type.setter
    def embedding_type(self, value: str):
        self._thread_local.embedding_type = value
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成文本嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            dim = self._dimension or 0
            return np.empty((0, dim), dtype=np.float32)
        
        num_texts = len(texts)
        num_batches = (num_texts + self.batch_size - 1) // self.batch_size
        logger.info(f"[Embedder] embed调用 | 模型: {self.model_name} | 类型: {self.embedding_type} | 文本数: {num_texts} | 批次数: {num_batches}")
        
        all_embeddings = []
        total_tokens_used = 0
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_idx = i // self.batch_size + 1
            last_err = None
            
            # 批次间等待，避免触发 RPM 限流
            if i > 0:
                time.sleep(0.2)
            
            from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS
            for _retry in range(API_RETRY_MAX_ATTEMPTS + 1):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens_used += response.usage.total_tokens
                    
                    if _retry > 0:
                        logger.info(f"[Embedder] 批次 {batch_idx}/{num_batches} 第{_retry+1}次重试成功")
                    
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if _retry < API_RETRY_MAX_ATTEMPTS:
                        wait = API_RETRY_INTERVAL_SECONDS
                        logger.warning(f"[Embedder] 批次 {batch_idx}/{num_batches} 失败 (第{_retry+1}/{API_RETRY_MAX_ATTEMPTS+1}次): {e}，{wait}s后重试...")
                        time.sleep(wait)
            
            if last_err is not None:
                logger.error(f"[Embedder] 批次 {batch_idx}/{num_batches} {API_RETRY_MAX_ATTEMPTS+1}次重试全部失败，程序终止！")
                raise last_err
        
        # 按嵌入类型分类统计 token 消耗
        if self.token_stats and total_tokens_used > 0:
            if self.embedding_type == "text":
                self.token_stats.add_text_embedding(total_tokens_used)
            elif self.embedding_type == "keyword":
                self.token_stats.add_keyword_embedding(total_tokens_used)
            elif self.embedding_type == "node":
                self.token_stats.add_node_embedding(total_tokens_used)
        
        logger.info(f"[Embedder] embed完成 | 模型: {self.model_name} | 消耗tokens: {total_tokens_used}")
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        if self._dimension is None:
            self._dimension = embeddings.shape[1]
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """生成查询嵌入向量"""
        return self.embed(query)[0]
    
    @property
    def dimension(self) -> int:
        """嵌入向量维度"""
        if self._dimension is None:
            test_embedding = self.embed("test")
            self._dimension = test_embedding.shape[1]
        return self._dimension


class LocalEmbedder(BaseEmbedder):
    """本地嵌入模型（基于sentence-transformers）"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace 模型名称或本地路径
            device: 设备（cuda/cpu），默认自动选择
            normalize: 是否 L2 归一化
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
        
        self.model_name = model_name
        self.normalize = normalize
        
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成文本嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        """生成查询嵌入向量"""
        # BGE 模型要求查询添加特定前缀
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.embed(query)[0]
    
    @property
    def dimension(self) -> int:
        """嵌入向量维度"""
        return self._dimension


class Embedder:
    """嵌入模型统一接口，屏蔽底层实现差异"""
    
    def __init__(
        self,
        embedder_type: str = "openai",
        **kwargs
    ):
        """
        Args:
            embedder_type: "openai" 或 "local"
            **kwargs: 传递给具体嵌入器的参数
        """
        self.embedder_type = embedder_type
        
        if embedder_type == "openai":
            self._embedder = OpenAIEmbedder(**kwargs)
        elif embedder_type == "local":
            self._embedder = LocalEmbedder(**kwargs)
        else:
            raise ValueError(f"不支持的嵌入器类型: {embedder_type}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """生成文本嵌入向量"""
        return self._embedder.embed(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """生成查询嵌入向量"""
        return self._embedder.embed_query(query)
    
    @property
    def dimension(self) -> int:
        """嵌入向量维度"""
        return self._embedder.dimension
    
    def set_embedding_type(self, embedding_type: str):
        """设置嵌入类型，用于分类统计 token"""
        if hasattr(self._embedder, 'embedding_type'):
            self._embedder.embedding_type = embedding_type