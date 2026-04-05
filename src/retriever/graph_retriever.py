# -*- coding: utf-8 -*-
"""
图谱检索器

基于向量相似度检索图谱节点，沿因果链扩展上下文。

Author: CongCongTian
"""

from typing import List, Optional

from src.models.schemas import (
    NodeType, RetrievalResult,
)
from src.graph.graph_store import GraphStore
from src.embedding.embedder import Embedder


class GraphRetriever:
    """图谱检索器
    
    向量相似度检索 + 因果链扩展 + 结构化上下文返回。
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        embedder: Embedder,
        expand_hops: int = 1,
        include_causal_chain: bool = True,
        max_chain_length: int = 5,
    ):
        """
        Args:
            graph_store: 图谱存储
            embedder: 嵌入模型
            expand_hops: 结果扩展跳数
            include_causal_chain: 是否包含因果链
            max_chain_length: 最大因果链长度
        """
        self.graph_store = graph_store
        self.embedder = embedder
        self.expand_hops = expand_hops
        self.include_causal_chain = include_causal_chain
        self.max_chain_length = max_chain_length
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        node_types: Optional[List[NodeType]] = None,
        expand_results: bool = True,
    ) -> List[RetrievalResult]:
        """检索与查询相关的图谱节点"""
        query_embedding = self.embedder.embed_query(query)
        
        results = self.graph_store.search_nodes(
            query_embedding,
            top_k=top_k * 2 if expand_results else top_k,
        )
        
        if node_types:
            results = [r for r in results if r.metadata
                       and NodeType(r.metadata.get("node_type", "")) in node_types]
        
        if not results:
            return []
        
        if expand_results and self.expand_hops > 0:
            results = self._expand_results(results, top_k)
        
        if self.include_causal_chain:
            results = self._add_causal_context(results)
        
        return results[:top_k]
    
    def _expand_results(
        self,
        results: List[RetrievalResult],
        target_count: int,
    ) -> List[RetrievalResult]:
        """扩展检索结果，纳入邻居节点"""
        seen_ids = {r.id for r in results}
        expanded = list(results)
        
        for result in results:
            if len(expanded) >= target_count:
                break
            
            neighbors = self.graph_store.graph.get_neighbors(
                result.id, direction="both"
            )
            
            for neighbor_id in neighbors:
                if neighbor_id in seen_ids:
                    continue
                if len(expanded) >= target_count:
                    break
                
                node = self.graph_store.graph.get_node(neighbor_id)
                if node:
                    neighbor_result = RetrievalResult(
                        id=neighbor_id,
                        text=node.name,
                        score=result.score * 0.7,
                        source_type="graph",
                        metadata={
                            "node_type": node.node_type.value,
                            "expanded_from": result.id,
                        }
                    )
                    expanded.append(neighbor_result)
                    seen_ids.add(neighbor_id)
        
        expanded.sort(key=lambda x: x.score, reverse=True)
        
        return expanded
    
    def _add_causal_context(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """为检索结果添加因果链上下文"""
        enhanced_results = []
        
        for result in results:
            forward_chain = self.graph_store.get_causal_chain(
                result.id, "forward", self.max_chain_length
            )
            backward_chain = self.graph_store.get_causal_chain(
                result.id, "backward", self.max_chain_length
            )
            
            chain_parts = []
            
            if len(backward_chain) > 1:
                backward_names = [
                    self.graph_store.graph.get_node(nid).name
                    for nid in backward_chain[:-1]  # 排除当前节点
                    if self.graph_store.graph.get_node(nid)
                ]
                if backward_names:
                    chain_parts.append(f"原因链: {' → '.join(backward_names)}")
            
            if len(forward_chain) > 1:
                forward_names = [
                    self.graph_store.graph.get_node(nid).name
                    for nid in forward_chain[1:]  # 排除当前节点
                    if self.graph_store.graph.get_node(nid)
                ]
                if forward_names:
                    chain_parts.append(f"后果链: {' → '.join(forward_names)}")
            
            context = self.graph_store.get_node_context(result.id)
            full_text = result.text
            if chain_parts:
                full_text += " | " + " | ".join(chain_parts)
            if context and context not in full_text:
                full_text += " | " + context
            
            enhanced_result = RetrievalResult(
                id=result.id,
                text=full_text,
                score=result.score,
                source_type="graph",
                metadata={
                    **result.metadata,
                    "forward_chain_length": len(forward_chain),
                    "backward_chain_length": len(backward_chain),
                }
            )
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
