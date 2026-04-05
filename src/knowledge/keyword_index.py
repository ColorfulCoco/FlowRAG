# -*- coding: utf-8 -*-
"""
关键词索引

存储关键词与知识块的关联关系，支持精确匹配和语义相似度检索。

Author: CongCongTian
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from src.embedding.embedder import Embedder
from src.config import LOG_VERBOSE


@dataclass
class KeywordEntry:
    """关键词条目"""
    keyword: str                                  # 关键词
    embedding: Optional[np.ndarray] = None        # 关键词嵌入向量
    chunk_ids: List[str] = field(default_factory=list)  # 关联的知识块ID
    frequency: int = 0                            # 出现频次
    
    def to_dict(self) -> Dict:
        """转换为字典（不含embedding）"""
        return {
            "keyword": self.keyword,
            "chunk_ids": self.chunk_ids,
            "frequency": self.frequency,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "KeywordEntry":
        """从字典创建"""
        return cls(
            keyword=data["keyword"],
            chunk_ids=data.get("chunk_ids", []),
            frequency=data.get("frequency", 0),
        )


class KeywordIndex:
    """关键词索引
    
    检索流程：query -> 提取关键词 -> 精确匹配 + 相似度检索 -> 关联知识块
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        persist_path: Optional[str] = None,
    ):
        """
        Args:
            embedder: 嵌入器，用于生成关键词嵌入
            persist_path: 持久化路径
        """
        self.embedder = embedder
        self.persist_path = Path(persist_path) if persist_path else None
        
        self._keywords: Dict[str, KeywordEntry] = {}
        self._chunk_to_keywords: Dict[str, List[str]] = defaultdict(list)  # 反向索引
        self._keyword_list: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        
        if self.persist_path:
            self._load()
    
    def add_keyword(
        self,
        keyword: str,
        chunk_id: str,
    ):
        """添加关键词与知识块的关联"""
        keyword = keyword.strip()
        if not keyword:
            return
        
        if keyword not in self._keywords:
            self._keywords[keyword] = KeywordEntry(keyword=keyword)
        
        entry = self._keywords[keyword]
        
        if chunk_id not in entry.chunk_ids:
            entry.chunk_ids.append(chunk_id)
            entry.frequency += 1
            
            if keyword not in self._chunk_to_keywords[chunk_id]:
                self._chunk_to_keywords[chunk_id].append(keyword)
    
    def add_chunk_keywords(
        self,
        chunk_id: str,
        keywords: List[str],
    ):
        """批量添加知识块的关键词关联"""
        for keyword in keywords:
            self.add_keyword(keyword, chunk_id)
    
    def build_embeddings(self, show_progress: bool = True, incremental: bool = True, pre_computed_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """为关键词生成嵌入向量，支持增量生成和复用已有嵌入
        
        Args:
            show_progress: 是否显示进度
            incremental: 是否增量（仅为新关键词生成）
            pre_computed_embeddings: MMR 过程中已生成的候选词嵌入，直接复用避免重复调 API
        """
        if not self.embedder:
            raise ValueError("未配置嵌入器，无法生成关键词嵌入")
        
        all_keywords = list(self._keywords.keys())
        if not all_keywords:
            print("[KeywordIndex] 警告: 没有关键词需要嵌入")
            return
        
        if incremental and self._embeddings is not None and len(self._keyword_list) > 0:
            old_keywords_set = set(self._keyword_list)
            new_keywords = [kw for kw in all_keywords if kw not in old_keywords_set]
            
            if not new_keywords:
                if LOG_VERBOSE:
                    print(f"[KeywordIndex] 没有新关键词，跳过嵌入生成")
                return
            
            reused_count, keywords_need_embed, emb_dict = self._split_by_pre_computed(
                new_keywords, pre_computed_embeddings
            )
            
            if LOG_VERBOSE:
                if reused_count > 0:
                    print(f"[KeywordIndex] 增量生成: {len(new_keywords)} 个新关键词"
                          f"（复用 {reused_count} 个已有嵌入，需调 API: {len(keywords_need_embed)} 个）")
                else:
                    print(f"[KeywordIndex] 增量生成: 为 {len(new_keywords)} 个新关键词生成嵌入...")
            
            if keywords_need_embed:
                self.embedder.set_embedding_type("keyword")
                api_embeddings = self.embedder.embed(keywords_need_embed)
                for kw, emb in zip(keywords_need_embed, api_embeddings):
                    emb_dict[kw] = emb
            
            new_embeddings = np.array([emb_dict[kw] for kw in new_keywords], dtype=np.float32)
            
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
            self._keyword_list.extend(new_keywords)
            
            for i, keyword in enumerate(new_keywords):
                self._keywords[keyword].embedding = new_embeddings[i]
            
            if LOG_VERBOSE:
                print(f"[KeywordIndex] 增量嵌入完成（新增 {len(new_keywords)} 个，共 {len(self._keyword_list)} 个）")
        
        else:
            reused_count, keywords_need_embed, emb_dict = self._split_by_pre_computed(
                all_keywords, pre_computed_embeddings
            )
            
            if LOG_VERBOSE:
                if reused_count > 0:
                    print(f"[KeywordIndex] 全量生成: {len(all_keywords)} 个关键词"
                          f"（复用 {reused_count} 个已有嵌入，需调 API: {len(keywords_need_embed)} 个）")
                else:
                    print(f"[KeywordIndex] 全量生成: 为 {len(all_keywords)} 个关键词生成嵌入...")
            
            if keywords_need_embed:
                self.embedder.set_embedding_type("keyword")
                api_embeddings = self.embedder.embed(keywords_need_embed)
                for kw, emb in zip(keywords_need_embed, api_embeddings):
                    emb_dict[kw] = emb
            
            embeddings = np.array([emb_dict[kw] for kw in all_keywords], dtype=np.float32)
            
            self._keyword_list = all_keywords
            self._embeddings = embeddings
            
            for i, keyword in enumerate(all_keywords):
                self._keywords[keyword].embedding = embeddings[i]
            
            if LOG_VERBOSE:
                print(f"[KeywordIndex] 全量嵌入完成")
        
        if self.persist_path:
            self._save()
    
    def _split_by_pre_computed(
        self,
        keywords: List[str],
        pre_computed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> tuple:
        """将关键词分为"可复用已有嵌入"和"需调 API"两组"""
        emb_dict = {}
        keywords_need_embed = []
        reused_count = 0
        
        for kw in keywords:
            if pre_computed_embeddings and kw in pre_computed_embeddings:
                emb_dict[kw] = pre_computed_embeddings[kw]
                reused_count += 1
            else:
                keywords_need_embed.append(kw)
        
        return reused_count, keywords_need_embed, emb_dict
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """通过嵌入向量检索相似关键词"""
        if self._embeddings is None or len(self._keyword_list) == 0:
            return []
        
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return []
        query_norm = query_embedding / q_norm
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings_norm = self._embeddings / norms
        
        similarities = np.dot(embeddings_norm, query_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= similarity_threshold:
                results.append((self._keyword_list[idx], score))
        
        return results
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """通过文本检索相似关键词"""
        if not self.embedder:
            raise ValueError("未配置嵌入器")
        
        query_embedding = self.embedder.embed([query_text])[0]
        return self.search_by_embedding(query_embedding, top_k, similarity_threshold)
    
    def get_chunks_by_keyword(self, keyword: str) -> List[str]:
        """获取关键词关联的知识块ID"""
        entry = self._keywords.get(keyword.strip())
        return entry.chunk_ids if entry else []
    
    def count(self) -> int:
        """获取关键词数量"""
        return len(self._keywords)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_keywords = len(self._keywords)
        total_chunks = len(self._chunk_to_keywords)
        total_links = sum(len(e.chunk_ids) for e in self._keywords.values())
        
        if total_keywords > 0:
            avg_chunks_per_keyword = total_links / total_keywords
        else:
            avg_chunks_per_keyword = 0
        
        if total_chunks > 0:
            avg_keywords_per_chunk = sum(
                len(kws) for kws in self._chunk_to_keywords.values()
            ) / total_chunks
        else:
            avg_keywords_per_chunk = 0
        
        return {
            "total_keywords": total_keywords,
            "total_chunks": total_chunks,
            "total_links": total_links,
            "avg_chunks_per_keyword": round(avg_chunks_per_keyword, 2),
            "avg_keywords_per_chunk": round(avg_keywords_per_chunk, 2),
            "has_embeddings": self._embeddings is not None,
        }
    
    def _save(self):
        """持久化索引到文件"""
        if not self.persist_path:
            return
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        meta_path = self.persist_path / "keyword_index.json"
        meta = {
            "keywords": {
                name: entry.to_dict() 
                for name, entry in self._keywords.items()
            },
            "keyword_list": self._keyword_list,
            "chunk_to_keywords": dict(self._chunk_to_keywords),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        if self._embeddings is not None:
            emb_path = self.persist_path / "keyword_embeddings.npz"
            np.savez(emb_path, embeddings=self._embeddings)
    
    def _load(self):
        """从文件恢复索引"""
        if not self.persist_path:
            return
        
        meta_path = self.persist_path / "keyword_index.json"
        emb_path = self.persist_path / "keyword_embeddings.npz"
        
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            for keyword, data in meta.get("keywords", {}).items():
                self._keywords[keyword] = KeywordEntry.from_dict(data)
            
            self._keyword_list = meta.get("keyword_list", [])
            
            chunk_to_keywords = meta.get("chunk_to_keywords", {})
            for chunk_id, keywords in chunk_to_keywords.items():
                self._chunk_to_keywords[chunk_id] = keywords
        
        if emb_path.exists():
            data = np.load(emb_path)
            self._embeddings = data["embeddings"]
            
            for i, keyword in enumerate(self._keyword_list):
                if keyword in self._keywords and i < len(self._embeddings):
                    self._keywords[keyword].embedding = self._embeddings[i]
    
