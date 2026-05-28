# -*- coding: utf-8 -*-
"""
结构上下文感知嵌入

基于残差结构增强，在节点嵌入中融合多跳拓扑上下文。
公式：h_v = Normalize(e_v + Σ_{d=1}^{K} γ^d · MeanAgg({e_u | u ∈ N_d(v)}))

Author: CongCongTian
"""

from typing import List, Dict, Optional, Set
import numpy as np

from src.models.schemas import GraphNode, KnowledgeGraph
from src.embedding.embedder import Embedder
from src.config import LOG_VERBOSE


class ContextEmbedder:
    """结构上下文感知嵌入器
    
    对图谱节点收集多跳邻居上下文，通过残差增强或加权融合
    将拓扑结构信息编码进嵌入向量。
    """
    
    def __init__(
        self,
        embedder: Embedder,
        context_weight: float = 0.3,
        max_context_length: int = 512,
        gamma: float = 0.5,
        max_hops: int = 2,
    ):
        """
        Args:
            embedder: 基础嵌入器
            context_weight: 上下文权重
            max_context_length: 上下文最大字符数
            gamma: 残差增强衰减因子
            max_hops: 最大跳数 K
        """
        self.embedder = embedder
        self.context_weight = context_weight
        self.max_context_length = max_context_length
        self.gamma = gamma
        self.max_hops = max_hops
        
        self._node_embedding_cache: Dict[str, np.ndarray] = {}  # 避免重复计算
    
    def _get_context_text(
        self,
        node: GraphNode,
        graph: KnowledgeGraph,
        direction: str = "both"
    ) -> str:
        """获取节点的结构上下文文本"""
        context_parts = []
        
        if direction in ["in", "both"] and node.predecessors:
            pred_texts = []
            for pred_id in node.predecessors[:3]:
                pred_node = graph.get_node(pred_id)
                if pred_node:
                    pred_texts.append(pred_node.name)
            if pred_texts:
                context_parts.append(f"前因: {', '.join(pred_texts)}")
        
        if direction in ["out", "both"] and node.successors:
            succ_texts = []
            for succ_id in node.successors[:3]:
                succ_node = graph.get_node(succ_id)
                if succ_node:
                    succ_texts.append(succ_node.name)
            if succ_texts:
                context_parts.append(f"后果: {', '.join(succ_texts)}")
        
        context_text = " | ".join(context_parts)
        
        if len(context_text) > self.max_context_length:
            context_text = context_text[:self.max_context_length] + "..."
        
        return context_text
    
    def embed_node_weighted(
        self,
        node: GraphNode,
        graph: KnowledgeGraph,
    ) -> np.ndarray:
        """加权融合方式生成上下文感知嵌入"""
        node_embedding = self.embedder.embed(node.name)[0]
        
        context_text = self._get_context_text(node, graph)
        if context_text:
            context_embedding = self.embedder.embed(context_text)[0]
            final_embedding = (
                (1 - self.context_weight) * node_embedding +
                self.context_weight * context_embedding
            )
            norm = np.linalg.norm(final_embedding)
            if norm > 1e-10:
                final_embedding = final_embedding / norm
            else:
                node_norm = np.linalg.norm(node_embedding)
                final_embedding = node_embedding / node_norm if node_norm > 1e-10 else node_embedding
        else:
            norm = np.linalg.norm(node_embedding)
            final_embedding = node_embedding / norm if norm > 1e-10 else node_embedding
        
        return final_embedding
    
    def _get_node_base_embedding(
        self,
        node: GraphNode,
        use_cache: bool = True
    ) -> np.ndarray:
        """获取节点的基础嵌入向量（带缓存）"""
        if use_cache and node.id in self._node_embedding_cache:
            return self._node_embedding_cache[node.id]
        
        embedding = self.embedder.embed(node.name)[0]
        
        if use_cache:
            self._node_embedding_cache[node.id] = embedding
        
        return embedding
    
    def _get_k_hop_neighbors(
        self,
        node: GraphNode,
        graph: KnowledgeGraph,
        max_hops: int
    ) -> Dict[int, List[str]]:
        """BFS 获取节点的多跳邻居，按跳数分组返回"""
        hop_neighbors: Dict[int, List[str]] = {}
        visited: Set[str] = {node.id}
        current_level: List[str] = [node.id]
        
        for hop in range(1, max_hops + 1):
            next_level: List[str] = []
            
            for nid in current_level:
                n = graph.get_node(nid)
                if n:
                    # 前驱 + 后继，考虑流程图的双向可达性
                    all_neighbors = list(set(n.predecessors + n.successors))
                    for neighbor_id in all_neighbors:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            next_level.append(neighbor_id)
            
            if next_level:
                hop_neighbors[hop] = next_level
            current_level = next_level
        
        return hop_neighbors
    
    def embed_node_residual(
        self,
        node: GraphNode,
        graph: KnowledgeGraph,
        gamma: Optional[float] = None,
        max_hops: Optional[int] = None,
    ) -> np.ndarray:
        """残差结构增强嵌入（推荐方法）
        
        解决同名节点（如不同阶段的"检查"）因文本相同而无法区分的问题。
        h_v = Normalize(e_v + Σ_{d=1}^{K} γ^d · MeanAgg({e_u | u ∈ N_d(v)}))
        
        Args:
            node: 目标节点
            graph: 知识图谱
            gamma: 衰减因子，默认使用实例配置值
            max_hops: 最大跳数 K，默认使用实例配置值
            
        Returns:
            归一化后的嵌入向量
        """
        gamma = gamma if gamma is not None else self.gamma
        max_hops = max_hops if max_hops is not None else self.max_hops
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        
        # 步骤1: 获取节点自身的原始嵌入 e_v
        node_embedding = self._get_node_base_embedding(node)
        
        # 步骤2: 获取多跳邻居并计算上下文贡献
        hop_neighbors = self._get_k_hop_neighbors(node, graph, max_hops)
        
        # 步骤3: 计算邻居上下文加权和
        # Σ_{d=1}^{K} γ^d · MeanAgg({e_u | u ∈ N_d(v)})
        context_sum = np.zeros_like(node_embedding)
        
        for hop, neighbor_ids in hop_neighbors.items():
            if not neighbor_ids:
                continue
            
            neighbor_embeddings = []
            for neighbor_id in neighbor_ids:
                neighbor_node = graph.get_node(neighbor_id)
                if neighbor_node:
                    neighbor_emb = self._get_node_base_embedding(neighbor_node)
                    neighbor_embeddings.append(neighbor_emb)
            
            if neighbor_embeddings:
                mean_embedding = np.mean(neighbor_embeddings, axis=0)
                
                # γ^d 衰减：距离越远影响越小
                decay_factor = gamma ** hop
                context_sum += decay_factor * mean_embedding
        
        # 步骤4: 残差连接 - 将上下文注入节点嵌入
        # h_v = e_v + context_sum
        final_embedding = node_embedding + context_sum
        
        norm = np.linalg.norm(final_embedding)
        if norm > 1e-10:
            final_embedding = final_embedding / norm
        else:
            node_norm = np.linalg.norm(node_embedding)
            if node_norm > 1e-10:
                final_embedding = node_embedding / node_norm
        
        return final_embedding
    
    def embed_all_nodes(
        self,
        graph: KnowledgeGraph,
        method: str = "residual",
        gamma: Optional[float] = None,
        max_hops: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """对图谱中所有节点生成上下文感知嵌入
        
        Args:
            graph: 知识图谱
            method: "residual"(残差增强) / "plain"(纯文本) / "weighted"(加权融合)
            gamma: 衰减因子，仅 residual 使用
            max_hops: 最大跳数，仅 residual 使用
            show_progress: 是否显示进度
            
        Returns:
            {节点ID: 归一化嵌入向量} 的映射
        """
        embeddings = {}
        
        self._node_embedding_cache.clear()
        
        # 批量生成基础 embedding，避免逐个调用 API
        if LOG_VERBOSE:
            print(f"      [步骤1/2] 批量生成所有节点的基础embedding...")
        
        # 并发场景下需要在调用前重新设置 embedding_type
        self.embedder.set_embedding_type("node")
        
        all_node_names = [node.name for node in graph.nodes.values()]
        all_base_embeddings = self.embedder.embed(all_node_names)
        
        for i, (node_id, node) in enumerate(graph.nodes.items()):
            self._node_embedding_cache[node_id] = all_base_embeddings[i]
        
        if LOG_VERBOSE:
            print(f"      [步骤2/2] 计算上下文增强embedding...")
        
        nodes = list(graph.nodes.items())
        
        if show_progress:
            try:
                from tqdm import tqdm
                nodes = tqdm(nodes, desc=f"生成节点嵌入({method})", unit="节点")
            except ImportError:
                pass
        
        for node_id, node in nodes:
            if method == "residual":
                embeddings[node_id] = self.embed_node_residual(
                    node, graph, gamma=gamma, max_hops=max_hops
                )
            elif method == "plain":
                emb = self._get_node_base_embedding(node)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings[node_id] = emb
            elif method == "weighted":
                embeddings[node_id] = self.embed_node_weighted(node, graph)
            else:
                raise ValueError(f"不支持的嵌入方法: {method}，可选: residual, plain, weighted")
        
        self._node_embedding_cache.clear()
        
        return embeddings
