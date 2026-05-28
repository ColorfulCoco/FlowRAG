# -*- coding: utf-8 -*-
"""
向量检索器

基于语义相似度从向量数据库中检索相关文本块。

Author: CongCongTian
"""

from typing import List, Optional

from src.models.schemas import RetrievalResult
from src.vector_store.vector_db import VectorDB
from src.embedding.embedder import Embedder


class VectorRetriever:
    """向量检索器，支持语义相似度检索和元数据过滤"""
    
    def __init__(
        self,
        vector_db: VectorDB,
        embedder: Embedder,
    ):
        self.vector_db = vector_db
        self.embedder = embedder
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_source: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """检索与查询语义相关的文本块"""
        query_embedding = self.embedder.embed_query(query)
        
        filter_metadata = None
        if filter_source:
            filter_metadata = {"source_file": filter_source}
        
        results = self.vector_db.search(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]
        
        return results[:top_k]
