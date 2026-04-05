# -*- coding: utf-8 -*-
"""
数据模型定义

图谱节点、边、文本块、检索结果等核心数据结构。

Author: CongCongTian
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np


class NodeType(Enum):
    """节点类型枚举"""
    EVENT = "event"           # 事件节点
    CONDITION = "condition"   # 条件节点
    CONSTRAINT = "constraint" # 约束节点
    EQUIPMENT = "equipment"   # 设备节点
    PARAMETER = "parameter"   # 参数节点


class EdgeType(Enum):
    """边类型枚举"""
    CAUSAL = "causal"         # 因果关系
    TRIGGERS = "triggers"     # 触发关系
    HAS_CONDITION = "has_condition"  # 具有条件
    RELATED = "related"       # 相关关系


@dataclass
class GraphNode:
    """图谱节点"""
    name: str                         # 节点名称（同时作为唯一标识）
    node_type: NodeType               # 节点类型
    properties: Dict[str, Any] = field(default_factory=dict)  # 附加属性
    source_tg: List[str] = field(default_factory=list)        # 来源TG文档
    embedding: Optional[np.ndarray] = None  # 节点嵌入向量
    
    # 结构上下文（用于上下文感知嵌入）
    predecessors: List[str] = field(default_factory=list)     # 前驱节点ID列表
    successors: List[str] = field(default_factory=list)       # 后继节点ID列表
    
    # 补充的陈述性知识
    enrichment_chunks: List[str] = field(default_factory=list)  # 关联的知识块ID
    enrichment_text: str = ""         # 拼接后的补充知识文本
    
    @property
    def id(self) -> str:
        """id 就是 name"""
        return self.name
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,  # name 就是唯一标识
            "node_type": self.node_type.value,
            "properties": self.properties,
            "source_tg": self.source_tg,
            "predecessors": self.predecessors,
            "successors": self.successors,
            "enrichment_chunks": self.enrichment_chunks,
            "enrichment_text": self.enrichment_text,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GraphNode":
        """从字典创建"""
        return cls(
            name=data.get("name", data.get("id", "")),  # 优先用 name，兼容旧数据用 id
            node_type=NodeType(data.get("node_type", "event")),
            properties=data.get("properties", {}),
            source_tg=data.get("source_tg", []),
            predecessors=data.get("predecessors", []),
            successors=data.get("successors", []),
            enrichment_chunks=data.get("enrichment_chunks", []),
            enrichment_text=data.get("enrichment_text", ""),
        )


@dataclass
class GraphEdge:
    """图谱边"""
    source_id: str                    # 源节点ID
    target_id: str                    # 目标节点ID
    edge_type: EdgeType               # 边类型
    relation_name: str = ""           # 关系名称（如"导致"、"触发"）
    confidence: float = 1.0           # 置信度
    properties: Dict[str, Any] = field(default_factory=dict)
    source_tg: str = ""               # 来源TG文档
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "relation_name": self.relation_name,
            "confidence": self.confidence,
            "properties": self.properties,
            "source_tg": self.source_tg,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GraphEdge":
        """从字典创建"""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data.get("edge_type", "causal")),
            relation_name=data.get("relation_name", ""),
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
            source_tg=data.get("source_tg", ""),
        )


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)  # 节点字典 {id: node}
    edges: List[GraphEdge] = field(default_factory=list)       # 边列表
    metadata: Dict[str, Any] = field(default_factory=dict)     # 元数据
    
    # 边索引（运行时自动构建，不序列化，用于 O(1) 查找边信息）
    _edge_index: Dict[Tuple[str, str], 'GraphEdge'] = field(default_factory=dict, init=False, repr=False)
    _outgoing_edges: Dict[str, List['GraphEdge']] = field(default_factory=dict, init=False, repr=False)
    _incoming_edges: Dict[str, List['GraphEdge']] = field(default_factory=dict, init=False, repr=False)
    
    def add_node(self, node: GraphNode):
        """添加节点"""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge):
        """添加边（自动去重：同一 source→target 只保留一条）"""
        key = (edge.source_id, edge.target_id)
        if key in self._edge_index:
            return
        self.edges.append(edge)
        if edge.source_id in self.nodes:
            if edge.target_id not in self.nodes[edge.source_id].successors:
                self.nodes[edge.source_id].successors.append(edge.target_id)
        if edge.target_id in self.nodes:
            if edge.source_id not in self.nodes[edge.target_id].predecessors:
                self.nodes[edge.target_id].predecessors.append(edge.source_id)
        self._edge_index[key] = edge
        self._outgoing_edges.setdefault(edge.source_id, []).append(edge)
        self._incoming_edges.setdefault(edge.target_id, []).append(edge)
    
    def get_edge(self, source_id: str, target_id: str) -> Optional['GraphEdge']:
        """O(1) 查找边：通过 (source_id, target_id) 获取边对象"""
        return self._edge_index.get((source_id, target_id))
    
    def get_outgoing_edges(self, node_id: str) -> List['GraphEdge']:
        """O(1) 获取节点的所有出边"""
        return self._outgoing_edges.get(node_id, [])
    
    def get_incoming_edges(self, node_id: str) -> List['GraphEdge']:
        """O(1) 获取节点的所有入边"""
        return self._incoming_edges.get(node_id, [])
    
    def _rebuild_edge_index(self):
        """从 edges 列表重建边索引（加载图谱后调用）"""
        self._edge_index = {}
        self._outgoing_edges = {}
        self._incoming_edges = {}
        for edge in self.edges:
            self._edge_index[(edge.source_id, edge.target_id)] = edge
            self._outgoing_edges.setdefault(edge.source_id, []).append(edge)
            self._incoming_edges.setdefault(edge.target_id, []).append(edge)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """获取邻居节点ID列表
        
        Args:
            node_id: 节点ID
            direction: "in"(入边)/"out"(出边)/"both"(双向)
        """
        neighbors = []
        node = self.nodes.get(node_id)
        if node:
            if direction in ["in", "both"]:
                neighbors.extend(node.predecessors)
            if direction in ["out", "both"]:
                neighbors.extend(node.successors)
        return list(set(neighbors))
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "KnowledgeGraph":
        """从字典创建"""
        graph = cls()
        graph.metadata = data.get("metadata", {})
        
        for nid, node_data in data.get("nodes", {}).items():
            graph.nodes[nid] = GraphNode.from_dict(node_data)
        
        for edge_data in data.get("edges", []):
            edge = GraphEdge.from_dict(edge_data)
            graph.edges.append(edge)
        
        # 从 edges 列表构建边索引，支持 O(1) 查找
        graph._rebuild_edge_index()
        
        return graph


@dataclass
class TextChunk:
    """文本块（用于陈述性知识存储）"""
    id: str                           # 唯一标识
    text: str                         # 文本内容
    source_file: str                  # 来源文件
    chunk_index: int                  # 在文档中的索引
    embedding: Optional[np.ndarray] = None  # 嵌入向量
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 关联信息
    related_node_ids: List[str] = field(default_factory=list)  # 关联的图谱节点ID
    
    # 知识分类
    knowledge_type: str = ""          # 知识类型: declarative/logical/hybrid
    entities: List[str] = field(default_factory=list)  # 提取的实体列表
    
    def to_dict(self) -> Dict:
        """转换为字典（不包含embedding）"""
        return {
            "id": self.id,
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "related_node_ids": self.related_node_ids,
            "knowledge_type": self.knowledge_type,
            "entities": self.entities,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TextChunk":
        """从字典创建"""
        return cls(
            id=data["id"],
            text=data["text"],
            source_file=data.get("source_file", ""),
            chunk_index=data.get("chunk_index", 0),
            metadata=data.get("metadata", {}),
            related_node_ids=data.get("related_node_ids", []),
            knowledge_type=data.get("knowledge_type", ""),
            entities=data.get("entities", []),
        )


@dataclass
class RetrievalResult:
    """单条检索结果"""
    id: str                           # 结果ID（节点ID或文本块ID）
    text: str = ""                         # 文本内容
    score: float = 0.0                # 相关度分数（默认0.0）
    source_type: str = ""             # 来源类型："graph" 或 "vector"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningPath:
    """推理路径 - 表示从锚点节点沿图谱的一条推理链"""
    anchor_node_id: str                       # 锚定节点ID
    anchor_score: float                       # 锚定节点的相似度分数
    path_nodes: List[str] = field(default_factory=list)  # 路径上的节点名称序列
    path_node_ids: List[str] = field(default_factory=list)  # 路径上的节点ID序列
    path_edges: List[str] = field(default_factory=list)  # 路径上的边类型序列
    path_score: float = 0.0                   # 路径整体得分
    context_relevance: float = 0.0            # 与查询意图的一致性得分
    is_pruned: bool = False                   # 是否被剪枝
    enrichment_chunk_ids: List[str] = field(default_factory=list)  # 路径节点挂载的知识chunk IDs（已去重）
    supplemental_knowledge: List[str] = field(default_factory=list)  # 补充的实体知识文本
    
    def to_dict(self) -> Dict:
        return {
            "anchor_node_id": self.anchor_node_id,
            "anchor_score": self.anchor_score,
            "path_nodes": self.path_nodes,
            "path_node_ids": self.path_node_ids,
            "path_edges": self.path_edges,
            "path_score": self.path_score,
            "context_relevance": self.context_relevance,
            "is_pruned": self.is_pruned,
            "enrichment_chunk_ids": self.enrichment_chunk_ids,
            "supplemental_knowledge": self.supplemental_knowledge,
        }
    


@dataclass
class ReasoningResult:
    """迭代式推理检索结果"""
    query: str                                # 原始查询
    
    # 检索结果（整合图谱和向量）
    graph_results: List[RetrievalResult] = field(default_factory=list)   # 图谱检索结果
    vector_results: List[RetrievalResult] = field(default_factory=list)  # 向量检索结果
    final_context: List[RetrievalResult] = field(default_factory=list)   # 最终上下文（已去重、已排序）
    
    # Agent 推理
    agent_answer: str = ""                    # Agent 推理得出的答案
    agent_reasoning: str = ""                 # Agent 的推理依据
    
    # 推理过程详情（可选，用于调试/分析）
    anchor_nodes: List[RetrievalResult] = field(default_factory=list)    # 锚点节点
    reasoning_paths: List[ReasoningPath] = field(default_factory=list)   # 推理路径
    
    # 统计信息
    total_iterations: int = 0                 # 总迭代次数
    explored_nodes_count: int = 0             # 探索的节点数
    graph_recall_count: int = 0               # 图谱召回数
    vector_recall_count: int = 0              # 向量召回数
    total_tokens: int = 0                     # 总 Token 消耗
    
    def get_context(self, max_items: int = 5) -> str:
        """获取用于LLM的上下文文本"""
        context_parts = []
        
        # 最终上下文
        if self.final_context:
            for i, result in enumerate(self.final_context[:max_items]):
                source_tag = "[图谱]" if result.source_type == "graph" else "[知识]"
                context_parts.append(f"{i+1}. {source_tag} {result.text}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "graph_results": [r.to_dict() for r in self.graph_results],
            "vector_results": [r.to_dict() for r in self.vector_results],
            "final_context": [r.to_dict() for r in self.final_context],
            "agent_answer": self.agent_answer,
            "agent_reasoning": self.agent_reasoning,
            "anchor_nodes": [r.to_dict() for r in self.anchor_nodes],
            "reasoning_paths": [p.to_dict() for p in self.reasoning_paths],
            "total_iterations": self.total_iterations,
            "explored_nodes_count": self.explored_nodes_count,
            "graph_recall_count": self.graph_recall_count,
            "vector_recall_count": self.vector_recall_count,
            "total_tokens": self.total_tokens,
        }


@dataclass
class HybridResult:
    """混合检索结果，实际数据在 reasoning_result 中"""
    query: str                        # 原始查询
    reasoning_result: Optional[ReasoningResult] = None
    fusion_strategy: str = "rrf"      # 融合策略：rrf, weighted, cascade
    
    # 代理属性，透传 reasoning_result
    @property
    def graph_results(self) -> List[RetrievalResult]:
        return self.reasoning_result.graph_results if self.reasoning_result else []
    
    @property
    def vector_results(self) -> List[RetrievalResult]:
        return self.reasoning_result.vector_results if self.reasoning_result else []
    
    @property
    def final_context(self) -> List[RetrievalResult]:
        return self.reasoning_result.final_context if self.reasoning_result else []
    
    @property
    def graph_recall_count(self) -> int:
        return self.reasoning_result.graph_recall_count if self.reasoning_result else 0
    
    @property
    def vector_recall_count(self) -> int:
        return self.reasoning_result.vector_recall_count if self.reasoning_result else 0
    
    def get_context(self, max_items: int = 5) -> str:
        """获取用于LLM的上下文文本"""
        if self.reasoning_result:
            return self.reasoning_result.get_context(max_items)
        return ""
    
    def to_dict(self) -> Dict:
        result: Dict[str, Any] = {
            "query": self.query,
            "fusion_strategy": self.fusion_strategy,
        }
        if self.reasoning_result:
            result["reasoning_result"] = self.reasoning_result.to_dict()
        return result