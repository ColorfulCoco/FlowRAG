# -*- coding: utf-8 -*-
"""
知识挂载器

遍历图谱节点，为每个节点挂载相似度最高的 TOP-K 知识块。

Author: CongCongTian
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.models.schemas import TextChunk, KnowledgeGraph, GraphNode
from src.vector_store.vector_db import VectorDB
from src.config import LOG_VERBOSE


@dataclass
class NodeMountingResult:
    """单个节点的挂载结果"""
    node_id: str
    node_name: str
    mounted_chunk_ids: List[str]  # 挂载的知识块ID
    scores: List[float]  # 对应的相似度分数


class KnowledgeMounter:
    """知识挂载器
    
    策略：节点找知识 —— 确保每个节点都关联到最相关的知识块。
    """
    
    def __init__(
        self,
        vector_db: VectorDB,
        top_k_chunks: int = 5,
    ):
        """
        Args:
            vector_db: 向量数据库
            top_k_chunks: 每个节点最多挂载的知识块数量
        """
        self.vector_db = vector_db
        self.top_k_chunks = top_k_chunks
        self.stats = {
            "total_nodes": 0,
            "total_chunks": 0,
            "nodes_with_chunks": 0,
            "total_mountings": 0,
        }
    
    def mount_knowledge(
        self,
        graph: KnowledgeGraph,
        chunks: Optional[List[TextChunk]] = None,
        show_progress: bool = True,
        debug: bool = False,
        only_new_nodes: bool = False,
    ) -> KnowledgeGraph:
        """执行知识挂载
        
        Args:
            graph: 知识图谱
            chunks: 知识块列表，为 None 时从 vector_db 获取
            show_progress: 是否显示进度
            debug: 调试模式
            only_new_nodes: 仅挂载 enrichment_chunks 为空的新节点
            
        Returns:
            更新后的知识图谱
        """
        if chunks is None:
            chunks = self.vector_db.get_all_chunks()
        
        if not chunks:
            print("[KnowledgeMounter] 警告: 没有知识块可供挂载")
            return graph
        
        if only_new_nodes:
            nodes_with_embedding = [
                (node_id, node) 
                for node_id, node in graph.nodes.items() 
                if node.embedding is not None and len(node.enrichment_chunks) == 0
            ]
            if LOG_VERBOSE and nodes_with_embedding:
                total_nodes = len([n for n in graph.nodes.values() if n.embedding is not None])
                skipped = total_nodes - len(nodes_with_embedding)
                print(f"[KnowledgeMounter] 增量模式: {len(nodes_with_embedding)} 个新节点需要挂载, {skipped} 个已挂载节点跳过")
        else:
            nodes_with_embedding = [
                (node_id, node) 
                for node_id, node in graph.nodes.items() 
                if node.embedding is not None
            ]
        
        if not nodes_with_embedding:
            if only_new_nodes:
                if LOG_VERBOSE:
                    print("[KnowledgeMounter] 所有节点都已挂载，无需处理")
            else:
                print("[KnowledgeMounter] 警告: 没有节点有嵌入向量，无法进行挂载")
            return graph
        
        self.stats["total_nodes"] = len(nodes_with_embedding)
        self.stats["total_chunks"] = len(chunks)
        
        chunk_embeddings, chunk_ids = self._collect_chunk_embeddings(chunks)
        
        if len(chunk_ids) == 0:
            print("[KnowledgeMounter] 警告: 没有知识块有嵌入向量")
            return graph
        
        if show_progress:
            try:
                from tqdm import tqdm
                nodes_iter = tqdm(nodes_with_embedding, desc="节点挂载", unit="node")
            except ImportError:
                nodes_iter = nodes_with_embedding
        else:
            nodes_iter = nodes_with_embedding
        
        for node_id, node in nodes_iter:
            result = self._mount_chunks_to_node(
                node_id, node, chunk_embeddings, chunk_ids, chunks, debug
            )
            
            if result.mounted_chunk_ids:
                self.stats["nodes_with_chunks"] += 1
                self.stats["total_mountings"] += len(result.mounted_chunk_ids)
        
        if LOG_VERBOSE:
            print(f"[KnowledgeMounter] 挂载完成: "
                  f"{self.stats['nodes_with_chunks']}/{self.stats['total_nodes']} 个节点已挂载, "
                  f"共 {self.stats['total_mountings']} 次挂载")
        
        return graph
    
    def _collect_chunk_embeddings(
        self,
        chunks: List[TextChunk],
    ) -> Tuple[np.ndarray, List[str]]:
        """收集知识块嵌入，构建归一化索引矩阵"""
        embeddings = []
        chunk_ids = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                chunk_ids.append(chunk.id)
        
        if embeddings:
            embeddings_matrix = np.vstack(embeddings)
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_matrix = embeddings_matrix / norms
            return embeddings_matrix, chunk_ids
        
        return np.array([]), []
    
    def _chunk_matches_source(self, chunk_id: str, source_tgs: set) -> bool:
        """检查 chunk ID 是否属于节点的来源文档（按文件名 stem 前缀匹配）"""
        for tg_id in source_tgs:
            if chunk_id.startswith(tg_id + "_") or chunk_id == tg_id:
                return True
        return False
    
    def _mount_chunks_to_node(
        self,
        node_id: str,
        node: GraphNode,
        chunk_embeddings: np.ndarray,
        chunk_ids: List[str],
        chunks: List[TextChunk],
        debug: bool = False,
    ) -> NodeMountingResult:
        """为单个节点挂载相似度最高的知识块，限定为来源相同的文档"""
        if len(chunk_ids) == 0 or node.embedding is None:
            return NodeMountingResult(
                node_id=node_id,
                node_name=node.name,
                mounted_chunk_ids=[],
                scores=[],
            )
        
        node_source_tgs = set(node.source_tg) if node.source_tg else set()
        
        if node_source_tgs:
            valid_indices = []
            for idx, cid in enumerate(chunk_ids):
                if self._chunk_matches_source(cid, node_source_tgs):
                    valid_indices.append(idx)
            
            if not valid_indices:
                return NodeMountingResult(
                    node_id=node_id,
                    node_name=node.name,
                    mounted_chunk_ids=[],
                    scores=[],
                )
        else:
            valid_indices = list(range(len(chunk_ids)))
        
        node_embedding = np.array(node.embedding)
        node_norm = np.linalg.norm(node_embedding)
        if node_norm > 0:
            node_embedding = node_embedding / node_norm
        
        valid_chunk_embeddings = chunk_embeddings[valid_indices]
        similarities = np.dot(valid_chunk_embeddings, node_embedding)
        
        top_k = min(self.top_k_chunks, len(valid_indices))
        local_top_indices = np.argsort(similarities)[::-1][:top_k]
        
        mounted_ids = []
        scores = []
        
        for local_idx in local_top_indices:
            original_idx = valid_indices[local_idx]
            score = similarities[local_idx]
           
            chunk_id = chunk_ids[original_idx]
            
            if chunk_id not in node.enrichment_chunks:
                node.enrichment_chunks.append(chunk_id)
                mounted_ids.append(chunk_id)
                scores.append(float(score))
            
        return NodeMountingResult(
            node_id=node_id,
            node_name=node.name,
            mounted_chunk_ids=mounted_ids,
            scores=scores,
        )
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
