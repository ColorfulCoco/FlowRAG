# -*- coding: utf-8 -*-
"""
图谱存储模块

将知识图谱持久化到本地 JSON 文件，内存中支持向量 + BM25 混合检索。

Author: CongCongTian
"""
# pyright: reportArgumentType=false

import logging
import re
import json
import threading
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np

from src.models.schemas import (
    KnowledgeGraph, NodeType, RetrievalResult
)
from src.config import LOG_VERBOSE

logger = logging.getLogger(__name__)

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class GraphStore:
    """图谱存储，节点 embedding 持久化到 .npz，图谱结构持久化到 JSON，检索在内存完成"""

    def __init__(
        self,
        namespace: str = "flow_rag",
        embedding_file: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        """初始化图谱存储"""
        if not re.match(r'^[a-zA-Z0-9_\u4e00-\u9fff]+$', namespace):
            raise ValueError(f"namespace 仅允许字母、数字、下划线和中文: {namespace!r}")

        self.namespace = namespace

        if persist_directory:
            self._persist_directory = Path(persist_directory)
        elif embedding_file:
            self._persist_directory = Path(embedding_file).parent
        else:
            self._persist_directory = None

        if embedding_file:
            self.embedding_file = embedding_file
        elif self._persist_directory:
            self.embedding_file = str(self._persist_directory / "node_embeddings.npz")
        else:
            self.embedding_file = None

        self._local_graph_file: Optional[str] = None
        if self._persist_directory:
            self._local_graph_file = str(self._persist_directory / "knowledge_graph.json")

        self._graph_cache: Optional[KnowledgeGraph] = None
        self._graph_load_lock = threading.Lock()
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._node_id_list: List[str] = []

        self._bm25_index = None
        self._tokenized_node_names: List[List[str]] = []
        self._bm25_lock = threading.Lock()

        self._vector_db = None
        self._embedder = None

        if self.embedding_file:
            self._load_embeddings_from_file()

    def close(self):
        """关闭（保留接口兼容性）"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== 存储操作 ====================

    def save_graph(self, graph: KnowledgeGraph, clear_existing: bool = False):
        """保存图谱到本地文件并更新内存缓存"""
        if clear_existing:
            self.clear()
        self._save_graph_to_local(graph)
        self._graph_cache = graph

    def _save_embeddings_to_file(self):
        """将节点 embedding 保存到 .npz 文件"""
        if not self.embedding_file or not self._node_embeddings:
            return

        embedding_path = Path(self.embedding_file)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)

        node_ids = list(self._node_embeddings.keys())
        embeddings_matrix = np.stack([self._node_embeddings[nid] for nid in node_ids])

        np.savez(
            self.embedding_file,
            node_ids=np.array(node_ids, dtype=object),
            embeddings=embeddings_matrix.astype(np.float32),
        )

    def _load_embeddings_from_file(self):
        """从 .npz 文件加载节点 embedding"""
        if not self.embedding_file:
            return

        embedding_path = Path(self.embedding_file)
        if not embedding_path.exists():
            return

        try:
            with np.load(self.embedding_file, allow_pickle=True) as data:
                node_ids = data['node_ids']
                embeddings = data['embeddings']

                self._node_embeddings = {
                    str(nid): emb for nid, emb in zip(node_ids, embeddings)
                }
            self._build_embedding_index()
        except Exception as e:
            logger.warning("加载 embedding 文件失败: %s", e)

    def _build_embedding_index(self):
        """构建归一化的 embedding 矩阵，加速余弦相似度计算"""
        if self._node_embeddings:
            self._node_id_list = list(self._node_embeddings.keys())
            self._embedding_matrix = np.stack(
                [self._node_embeddings[nid] for nid in self._node_id_list]
            ).astype(np.float32)
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._embedding_matrix = self._embedding_matrix / norms

    # ==================== 加载操作 ====================

    def load_graph(self) -> KnowledgeGraph:
        """从本地 JSON 文件加载图谱，文件不存在则返回空图谱"""
        if self._local_graph_file:
            local_path = Path(self._local_graph_file)
            if local_path.exists():
                try:
                    graph = self._load_graph_from_local()
                    if LOG_VERBOSE:
                        print(f"[GraphStore] 从本地文件加载图谱: {len(graph.nodes)} 节点, {len(graph.edges)} 边 "
                              f"({local_path.name})")
                    if not self._node_embeddings and self.embedding_file:
                        self._load_embeddings_from_file()
                    self._graph_cache = graph
                    return graph
                except Exception as e:
                    logger.warning("本地文件加载失败: %s，返回空图谱", e)

        graph = KnowledgeGraph()
        self._graph_cache = graph
        return graph

    def _save_graph_to_local(self, graph: KnowledgeGraph):
        """将图谱序列化到 knowledge_graph.json（embedding 单独保存在 .npz 中）"""
        if not self._local_graph_file:
            return

        try:
            local_path = Path(self._local_graph_file)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)

            if LOG_VERBOSE:
                print(f"[GraphStore] 图谱已保存到本地: {local_path.name} "
                      f"({len(graph.nodes)} 节点, {len(graph.edges)} 边)")
        except Exception as e:
            logger.warning("保存本地图谱失败: %s", e)

    def _load_graph_from_local(self) -> KnowledgeGraph:
        """从 knowledge_graph.json 反序列化图谱"""
        with open(self._local_graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return KnowledgeGraph.from_dict(data)

    @property
    def graph(self) -> KnowledgeGraph:
        """获取图谱（double-checked locking，线程安全）"""
        if self._graph_cache is not None:
            return self._graph_cache
        with self._graph_load_lock:
            if self._graph_cache is None:
                self._graph_cache = self.load_graph()
            return self._graph_cache

    # ==================== 查询操作 ====================

    def search_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """基于向量余弦相似度搜索节点"""
        if self._embedding_matrix is None or len(self._node_id_list) == 0:
            return []

        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return []
        query_embedding = (query_embedding / q_norm).astype(np.float32)

        scores = np.dot(self._embedding_matrix, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            node_id = self._node_id_list[idx]
            node = self.graph.get_node(node_id)
            if node:
                text_parts = [node.name]
                if node.enrichment_chunks:
                    enrichment_text = self.get_enrichment_text_by_chunks(node.enrichment_chunks)
                    if enrichment_text:
                        text_parts.append(enrichment_text)

                results.append(RetrievalResult(
                    id=node_id,
                    text=" | ".join(text_parts),
                    score=float(scores[idx]),
                    source_type="graph",
                    metadata={
                        "node_type": node.node_type.value,
                        "source_tg": node.source_tg,
                        "predecessors": node.predecessors,
                        "successors": node.successors,
                        "enrichment_chunks": node.enrichment_chunks,
                    }
                ))

        return results

    # ==================== BM25 混合检索 ====================

    def _tokenize(self, text: str) -> List[str]:
        """分词，优先使用 jieba，降级为正则分割"""
        if not text:
            return []

        text = text.lower()

        if JIEBA_AVAILABLE:
            tokens = list(jieba.cut(text))
        else:
            tokens = re.split(r'[\s\n\r\t，。！？；：、（）【】《》""'']+', text)

        filtered = []
        for token in tokens:
            token = token.strip()
            if len(token) >= 1:
                filtered.append(token)

        return filtered

    def build_bm25_index(self):
        """构建 BM25 索引（索引内容: 节点名称 + source_tg + keywords）"""
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 未安装，BM25 节点检索不可用")
            return

        if not self._node_id_list:
            return

        self._tokenized_node_names = []
        for node_id in self._node_id_list:
            node = self.graph.get_node(node_id)
            if node:
                text = node.name
                if node.source_tg:
                    text += " " + " ".join(node.source_tg)
                keywords = node.properties.get("keywords", [])
                if keywords:
                    text += " " + " ".join(keywords)
                tokens = self._tokenize(text)
                self._tokenized_node_names.append(tokens)
            else:
                self._tokenized_node_names.append([])

        self._bm25_index = BM25Okapi(self._tokenized_node_names)
        if LOG_VERBOSE:
            print(f"[BM25] 节点索引构建完成，共 {len(self._tokenized_node_names)} 个节点（含 keywords）")

    def search_nodes_bm25(
        self,
        query: str,
        top_k: int = 10,
        node_types: Optional[List[NodeType]] = None,
    ) -> List[RetrievalResult]:
        """基于 BM25 关键词搜索节点"""
        if not BM25_AVAILABLE:
            return []

        if self._bm25_index is None:
            with self._bm25_lock:
                if self._bm25_index is None:
                    self.build_bm25_index()
                    if self._bm25_index is None:
                        return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25_index.get_scores(query_tokens)

        if node_types:
            for i, nid in enumerate(self._node_id_list):
                node = self.graph.get_node(nid)
                if node and node.node_type not in node_types:
                    scores[i] = -1

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            node_id = self._node_id_list[idx]
            node = self.graph.get_node(node_id)
            if node:
                text_parts = [node.name]
                if node.enrichment_chunks:
                    enrichment_text = self.get_enrichment_text_by_chunks(node.enrichment_chunks)
                    if enrichment_text:
                        text_parts.append(enrichment_text)

                results.append(RetrievalResult(
                    id=node_id,
                    text=" | ".join(text_parts),
                    score=float(scores[idx]),
                    source_type="bm25",
                    metadata={
                        "node_type": node.node_type.value,
                        "source_tg": node.source_tg,
                        "predecessors": node.predecessors,
                        "successors": node.successors,
                        "enrichment_chunks": node.enrichment_chunks,
                    }
                ))

        return results

    def search_nodes_vector_then_keywords(
        self,
        system_device: str,
        query_embedding: np.ndarray,
        origin_query: str,
        top_k: int = 10,
        node_types: Optional[List[NodeType]] = None,
        recall_multiplier: int = 2,
    ) -> List[RetrievalResult]:
        """向量 + BM25 多路召回 + RRF 融合检索"""
        return self._search_multi_signal_fusion(
            system_device=system_device,
            query_embedding=query_embedding,
            origin_query=origin_query,
            top_k=top_k,
            node_types=node_types,
            recall_multiplier=recall_multiplier,
        )

    def _search_multi_signal_fusion(
        self,
        system_device: str,
        query_embedding: np.ndarray,
        origin_query: str,
        top_k: int = 10,
        node_types: Optional[List[NodeType]] = None,
        recall_multiplier: int = 3,
        rrf_k: int = 60,
    ) -> List[RetrievalResult]:
        """向量 + BM25 多路召回经 RRF 融合，再以 system_device keyword 微调排序"""
        recall_k = top_k * recall_multiplier

        vector_candidates = self.search_nodes(query_embedding, recall_k)
        bm25_candidates = self.search_nodes_bm25(origin_query, recall_k, node_types)

        if LOG_VERBOSE:
            print(f"[多信号融合] 向量召回: {len(vector_candidates)}, BM25 召回: {len(bm25_candidates)}")

        rrf_scores: Dict[str, float] = {}
        node_info: Dict[str, Dict] = {}

        for rank, candidate in enumerate(vector_candidates):
            nid = candidate.id
            rrf_scores[nid] = rrf_scores.get(nid, 0) + 1.0 / (rrf_k + rank + 1)
            if nid not in node_info:
                node_info[nid] = {
                    "vector_rank": rank + 1,
                    "bm25_rank": None,
                    "vector_score": candidate.score,
                    "bm25_score": 0.0,
                    "candidate": candidate,
                }
            node_info[nid]["vector_rank"] = rank + 1
            node_info[nid]["vector_score"] = candidate.score

        for rank, candidate in enumerate(bm25_candidates):
            nid = candidate.id
            rrf_scores[nid] = rrf_scores.get(nid, 0) + 1.0 / (rrf_k + rank + 1)
            if nid not in node_info:
                node_info[nid] = {
                    "vector_rank": None,
                    "bm25_rank": rank + 1,
                    "vector_score": 0.0,
                    "bm25_score": candidate.score,
                    "candidate": candidate,
                }
            node_info[nid]["bm25_rank"] = rank + 1
            node_info[nid]["bm25_score"] = candidate.score

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        keyword_boost: Dict[str, float] = {}
        if system_device and self._embedder:
            try:
                sd_embedding = self._embedder.embed_query(system_device)
                sd_embedding = np.array(sd_embedding, dtype=np.float32)
                sd_norm = sd_embedding / (np.linalg.norm(sd_embedding) + 1e-8)

                for nid in rrf_scores:
                    node = self.graph.get_node(nid)
                    if node:
                        keywords = node.properties.get("keywords", [])
                        if keywords:
                            try:
                                kw_text = " ".join(keywords)
                                kw_embedding = self._embedder.embed_query(kw_text)
                                kw_norm = np.array(kw_embedding, dtype=np.float32)
                                kw_norm = kw_norm / (np.linalg.norm(kw_norm) + 1e-8)
                                sim = float(np.dot(sd_norm, kw_norm))
                                keyword_boost[nid] = max(0, sim)
                            except Exception as e:
                                logging.getLogger(__name__).debug(f"keyword boost计算失败 ({nid}): {e}")
            except Exception as e:
                logger.debug("system_device embedding 失败: %s，跳过 keyword 微调", e)

        # keyword_boost 权重低于 RRF，仅做微调
        final_scores: List[tuple] = []
        keyword_boost_weight = 0.1
        for nid, rrf_score in sorted_candidates:
            kb = keyword_boost.get(nid, 0.0)
            final_score = rrf_score + keyword_boost_weight * kb
            final_scores.append((final_score, rrf_score, kb, nid))

        final_scores.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievalResult] = []
        for final_score, rrf_score, kb, nid in final_scores[:top_k]:
            node = self.graph.get_node(nid)
            if not node:
                continue

            info = node_info[nid]
            keywords = node.properties.get("keywords", [])

            metadata = {
                "node_type": node.node_type.value,
                "source_tg": node.source_tg,
                "predecessors": node.predecessors,
                "successors": node.successors,
                "enrichment_chunks": node.enrichment_chunks,
                "keywords": keywords,
                "rrf_score": float(rrf_score),
                "vector_rank": info["vector_rank"],
                "bm25_rank": info["bm25_rank"],
                "vector_score": float(info["vector_score"]),
                "bm25_score": float(info["bm25_score"]),
                "keyword_boost": float(kb),
            }

            text_parts = [node.name]
            if node.enrichment_chunks:
                enrichment_text = self.get_enrichment_text_by_chunks(node.enrichment_chunks)
                if enrichment_text:
                    text_parts.append(enrichment_text)

            result = RetrievalResult(
                id=nid,
                text=" | ".join(text_parts),
                score=float(final_score),
                source_type="multi_signal_fusion",
                metadata=metadata,
            )
            results.append(result)

        if LOG_VERBOSE:
            for i, r in enumerate(results[:5]):
                meta = r.metadata or {}
                print(f"  [多信号融合] {i+1}: {r.id} rrf={meta.get('rrf_score', 0):.4f} "
                      f"v_rank={meta.get('vector_rank')} bm25_rank={meta.get('bm25_rank')}")

        return results

    def get_causal_chain(
        self,
        node_id: str,
        direction: str = "forward",
        max_depth: int = 5,
    ) -> List[str]:
        """DFS 遍历因果链，返回节点 ID 列表"""
        return self._get_causal_chain_in_memory(node_id, direction, max_depth)

    def _get_causal_chain_in_memory(
        self,
        node_id: str,
        direction: str = "forward",
        max_depth: int = 5,
    ) -> List[str]:
        """内存 DFS 遍历，返回从 node_id 出发沿指定方向的最长链"""
        graph = self._graph_cache
        if graph is None:
            return [node_id]

        start_node = graph.get_node(node_id)
        if not start_node:
            return [node_id]

        best_chain: List[str] = [node_id]

        def dfs(current_id: str, path: List[str], depth: int):
            nonlocal best_chain
            if depth >= max_depth:
                if len(path) > len(best_chain):
                    best_chain = list(path)
                return

            current_node = graph.get_node(current_id)
            if not current_node:
                if len(path) > len(best_chain):
                    best_chain = list(path)
                return

            neighbors = current_node.successors if direction == "forward" else current_node.predecessors

            if not neighbors:
                if len(path) > len(best_chain):
                    best_chain = list(path)
                return

            visited = set(path)
            found_unvisited = False
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    found_unvisited = True
                    path.append(neighbor_id)
                    dfs(neighbor_id, path, depth + 1)
                    path.pop()

            if not found_unvisited:
                if len(path) > len(best_chain):
                    best_chain = list(path)

        dfs(node_id, [node_id], 0)
        return best_chain

    def set_vector_db(self, vector_db):
        """设置向量库引用，用于根据 chunk ID 获取文本"""
        self._vector_db = vector_db

    def set_embedder(self, embedder):
        """设置 Embedder 引用"""
        self._embedder = embedder

    def set_embedding_file(self, embedding_file: str):
        """设置 embedding 文件路径并尝试加载"""
        self.embedding_file = embedding_file
        if embedding_file:
            cache_dir = Path(embedding_file).parent
            self._local_graph_file = str(cache_dir / "knowledge_graph.json")
            self._persist_directory = cache_dir
        self._load_embeddings_from_file()

    def get_enrichment_text_by_chunks(
        self,
        chunk_ids: List[str],
        max_length: int = 300,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """根据 chunk ID 列表从向量库获取文本，可按 query 相关性排序"""
        if not chunk_ids or not self._vector_db:
            return ""

        chunks = []
        for chunk_id in chunk_ids:
            chunk = self._vector_db.get_chunk(chunk_id)
            if chunk:
                chunks.append(chunk)

        if not chunks:
            return ""

        if query and self._embedder:
            try:
                query_embedding = self._embedder.embed_query(query)

                chunk_scores = []
                for chunk in chunks:
                    if chunk.embedding is not None:
                        norm_prod = np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                        if norm_prod > 1e-9:
                            similarity = float(np.dot(query_embedding, chunk.embedding) / norm_prod)
                            chunk_scores.append((chunk, similarity))
                    else:
                        chunk_scores.append((chunk, 0.0))

                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                if top_k is not None:
                    chunk_scores = chunk_scores[:top_k]

                chunks = [c for c, _ in chunk_scores]
            except Exception as e:
                logger.debug("按相似度排序失败: %s", e)
                if top_k is not None:
                    chunks = chunks[:top_k]
        elif top_k is not None:
            chunks = chunks[:top_k]

        texts = []
        for chunk in chunks:
            text = chunk.text[:max_length]
            if len(chunk.text) > max_length:
                text += "..."
            texts.append(text)

        return " | ".join(texts)

    def get_node_context(
        self,
        node_id: str,
        include_enrichment: bool = True,
        query: Optional[str] = None,
    ) -> str:
        """获取节点上下文文本，可包含 enrichment chunk 内容"""
        node = self.graph.get_node(node_id)
        if not node:
            return ""

        context_parts = [f"【事件】{node.name}"]

        if include_enrichment and node.enrichment_chunks:
            enrichment_text = self.get_enrichment_text_by_chunks(
                node.enrichment_chunks,
                query=query,
                top_k=1 if query else None
            )
            if enrichment_text:
                context_parts.append(f"【相关知识】{enrichment_text}")

        return "\n".join(context_parts)

    # ==================== 管理操作 ====================

    def clear(self, delete_files: bool = False):
        """清除内存缓存，可选删除本地文件"""
        self._graph_cache = None
        self._node_embeddings.clear()
        self._embedding_matrix = None
        self._node_id_list.clear()
        self._bm25_index = None
        self._tokenized_node_names = []

        if delete_files:
            if self._local_graph_file:
                p = Path(self._local_graph_file)
                if p.exists():
                    p.unlink()
            if self.embedding_file:
                p = Path(self.embedding_file)
                if p.exists():
                    p.unlink()

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if self._graph_cache is not None:
            graph = self._graph_cache
            node_count = len(graph.nodes)
            edge_count = len(graph.edges)
            node_types: Dict[str, int] = {}
            for node in graph.nodes.values():
                nt = node.node_type.value
                node_types[nt] = node_types.get(nt, 0) + 1
            return {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "node_types": node_types,
                "has_embeddings": len(self._node_embeddings) > 0,
                "namespace": self.namespace,
            }

        return {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "has_embeddings": len(self._node_embeddings) > 0,
            "namespace": self.namespace,
        }

    def set_node_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """设置节点嵌入，同步更新内存索引、图谱缓存和本地文件"""
        self._node_embeddings = embeddings
        self._build_embedding_index()

        if self._graph_cache:
            for node_id, embedding in embeddings.items():
                if node_id in self._graph_cache.nodes:
                    self._graph_cache.nodes[node_id].embedding = embedding

        if self.embedding_file:
            self._save_embeddings_to_file()

