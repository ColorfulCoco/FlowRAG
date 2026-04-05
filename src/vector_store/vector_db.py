# -*- coding: utf-8 -*-
"""
向量数据库封装

统一封装 FAISS 和 ChromaDB 后端，提供增删查和持久化接口。

Author: CongCongTian
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from src.models.schemas import TextChunk, RetrievalResult


class VectorDB:
    """向量数据库，支持 FAISS（高性能）和 ChromaDB（轻量、元数据过滤）两种后端"""
    
    def __init__(
        self,
        backend: str = "faiss",
        dimension: int = 1536,
        index_type: str = "IVFFlat",  # FAISS索引类型
        persist_directory: Optional[str] = None,
        collection_name: str = "flowrag",
    ):
        """
        Args:
            backend: "faiss" 或 "chroma"
            dimension: 向量维度
            index_type: FAISS 索引类型（Flat / IVFFlat / HNSW）
            persist_directory: 持久化目录
            collection_name: ChromaDB 集合名称
        """
        self.backend = backend
        self.dimension = dimension
        self.index_type = index_type
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self._id_to_chunk: Dict[str, TextChunk] = {}
        self._id_list: List[str] = []
        
        if backend == "faiss":
            self._init_faiss()
        elif backend == "chroma":
            self._init_chroma()
        else:
            raise ValueError(f"不支持的后端: {backend}")
    
    def _init_faiss(self):
        """初始化 FAISS 索引，支持 Flat / IVFFlat / HNSW"""
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装 FAISS: pip install faiss-cpu 或 faiss-gpu")
        
        self.faiss = faiss
        
        if self.index_type == "Flat":
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self._is_trained = False
        elif self.index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self._index = faiss.IndexFlatIP(self.dimension)
        
        if self.persist_directory:
            self._load_faiss_index()
    
    def _init_chroma(self):
        """初始化 ChromaDB 客户端和集合"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("请安装 ChromaDB: pip install chromadb")
        
        if self.persist_directory:
            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            ))
        else:
            self._client = chromadb.Client()
        
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _load_faiss_index(self):
        """从持久化目录加载 FAISS 索引和元数据"""
        if not self.persist_directory:
            return
        
        index_path = Path(self.persist_directory) / "faiss_index.bin"
        meta_path = Path(self.persist_directory) / "faiss_meta.json"
        
        if index_path.exists() and meta_path.exists():
            self._index = self.faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self._id_list = meta.get("id_list", [])
                for chunk_data in meta.get("chunks", []):
                    chunk = TextChunk.from_dict(chunk_data)
                    self._id_to_chunk[chunk.id] = chunk
    
    def _save_faiss_index(self):
        """将 FAISS 索引和元数据写入磁盘"""
        if not self.persist_directory:
            return
        
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        index_path = Path(self.persist_directory) / "faiss_index.bin"
        meta_path = Path(self.persist_directory) / "faiss_meta.json"
        
        self.faiss.write_index(self._index, str(index_path))
        
        meta = {
            "id_list": self._id_list,
            "chunks": [chunk.to_dict() for chunk in self._id_to_chunk.values()],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    def add(
        self,
        chunks: List[TextChunk],
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        添加文本块及其嵌入向量

        Args:
            chunks: 文本块列表
            embeddings: 对应的嵌入向量，shape (n, dimension)
        """
        if embeddings is None:
            raise ValueError("必须提供嵌入向量")
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float32)
        
        if self.backend == "faiss":
            if hasattr(self, '_is_trained') and not self._is_trained:
                existing_count = self._index.ntotal if self._index else 0
                total_vectors = existing_count + len(chunks)
                if total_vectors >= 100:
                    all_emb = embeddings
                    if existing_count > 0:
                        old_emb = np.array([self._index.reconstruct(i) for i in range(existing_count)], dtype=np.float32)
                        all_emb = np.vstack([old_emb, embeddings])
                    self._index = self.faiss.IndexIVFFlat(
                        self.faiss.IndexFlatIP(self.dimension),
                        self.dimension, min(16, total_vectors // 8),
                        self.faiss.METRIC_INNER_PRODUCT
                    )
                    self._index.train(all_emb)
                    self._index.add(all_emb)
                    self._is_trained = True
                    for i, chunk in enumerate(chunks):
                        chunk.embedding = embeddings[i]
                        self._id_to_chunk[chunk.id] = chunk
                        self._id_list.append(chunk.id)
                    return
                else:
                    old_count = self._index.ntotal if self._index else 0
                    flat_index = self.faiss.IndexFlatIP(self.dimension)
                    if old_count > 0:
                        old_emb = np.array([self._index.reconstruct(i) for i in range(old_count)], dtype=np.float32)
                        flat_index.add(old_emb)
                    self._index = flat_index
                    self._is_trained = True
            
            self._index.add(embeddings)
            
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                self._id_to_chunk[chunk.id] = chunk
                self._id_list.append(chunk.id)
            
            self._save_faiss_index()
        
        elif self.backend == "chroma":
            self._collection.add(
                ids=[chunk.id for chunk in chunks],
                embeddings=embeddings.tolist(),
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
            )
            
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                self._id_to_chunk[chunk.id] = chunk
                self._id_list.append(chunk.id)
    
    def _match_filter(self, chunk: TextChunk, filter_metadata: Optional[Dict]) -> bool:
        """检查文本块的 metadata 是否满足过滤条件"""
        if not filter_metadata:
            return True
        
        for key, value in filter_metadata.items():
            chunk_value = chunk.metadata.get(key)
            if chunk_value != value:
                return False
        return True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        向量相似度检索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量，None 表示全部返回
            filter_metadata: 元数据过滤条件
        """
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return []
        query_embedding = (query_embedding / q_norm).astype(np.float32).reshape(1, -1)
        
        results = []
        
        if self.backend == "faiss":
            if self._index.ntotal == 0:
                return []
            
            if top_k is None:
                actual_k = self._index.ntotal
            else:
                # 有过滤条件时多取 5 倍，保证过滤后仍有足够结果
                search_k = top_k * 5 if filter_metadata else top_k
                actual_k = min(search_k, self._index.ntotal)
            scores, indices = self._index.search(query_embedding, actual_k)
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._id_list):
                    continue
                
                chunk_id = self._id_list[idx]
                chunk = self._id_to_chunk.get(chunk_id)
                
                if chunk and self._match_filter(chunk, filter_metadata):
                    results.append(RetrievalResult(
                        id=chunk.id,
                        text=chunk.text,
                        score=float(score),
                        source_type="vector",
                        metadata={
                            "source_file": chunk.source_file,
                            "chunk_index": chunk.chunk_index,
                            **chunk.metadata,
                        }
                    ))
                    if top_k is not None and len(results) >= top_k:
                        break
        
        elif self.backend == "chroma":
            # ChromaDB 不支持返回全部，用大数字兜底
            chroma_n_results = top_k if top_k is not None else 10000
            query_result = self._collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=chroma_n_results,
                where=filter_metadata,
            )
            
            if query_result["ids"] and query_result["ids"][0]:
                for i, chunk_id in enumerate(query_result["ids"][0]):
                    chunk = self._id_to_chunk.get(chunk_id)
                    distance = query_result["distances"][0][i] if query_result["distances"] else 0
                    score = 1 - distance
                    
                    if chunk:
                        results.append(RetrievalResult(
                            id=chunk.id,
                            text=chunk.text,
                            score=score,
                            source_type="vector",
                            metadata={
                                "source_file": chunk.source_file,
                                "chunk_index": chunk.chunk_index,
                                **chunk.metadata,
                            }
                        ))
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """根据 ID 获取文本块"""
        return self._id_to_chunk.get(chunk_id)
    
    def get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """从 FAISS 索引重建嵌入向量，不产生 API 调用"""
        if self.backend != "faiss":
            chunk = self._id_to_chunk.get(chunk_id)
            return chunk.embedding if chunk and chunk.embedding is not None else None
        
        if chunk_id not in self._id_to_chunk:
            return None
        try:
            idx = self._id_list.index(chunk_id)
            return self._index.reconstruct(idx)
        except (ValueError, RuntimeError):
            return None
    
    def get_all_chunks(self) -> List[TextChunk]:
        """获取所有文本块"""
        return list(self._id_to_chunk.values())
    
    def count(self) -> int:
        """返回当前索引中的向量数量"""
        if self.backend == "faiss":
            return self._index.ntotal
        elif self.backend == "chroma":
            return self._collection.count()
        return 0
    
    def clear(self):
        """清空索引和所有元数据"""
        if self.backend == "faiss":
            self._init_faiss()
        elif self.backend == "chroma":
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        self._id_to_chunk.clear()
        self._id_list.clear()
    
    