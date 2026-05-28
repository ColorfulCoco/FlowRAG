# -*- coding: utf-8 -*-
"""
Mermaid 解析与图谱生成

提供 Mermaid 代码解析（MermaidParser）和 LLM 驱动的图谱生成（GraphGenerator / MermaidOnlyGenerator）。

Author: CongCongTian
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.models.schemas import GraphNode, GraphEdge, KnowledgeGraph, NodeType, EdgeType


def _extract_json_robust(content: str) -> Optional[dict]:
    """从 LLM 响应中提取 JSON（代码块 -> 直接解析 -> 括号匹配）"""
    for match in re.finditer(r'```(?:json)?\s*([\s\S]*?)```', content, re.IGNORECASE):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    for m in re.finditer(r'\{', content):
        depth, end = 0, -1
        for i in range(m.start(), len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            try:
                return json.loads(content[m.start():end])
            except json.JSONDecodeError:
                continue

    return None


def _extract_mermaid_codes(content: str) -> 'MermaidCodes':
    """从 LLM 输出中提取 Mermaid 代码（代码块 -> 行扫描 -> 单反引号 -> 裸关键字）"""
    result = MermaidCodes()

    code_block_pattern = re.compile(r'```(?:mermaid)?\s*([\s\S]*?)```', re.IGNORECASE)
    for match in code_block_pattern.findall(content):
        match_stripped = match.strip()
        match_lower = match_stripped.lower()
        if match_lower.startswith('flowchart') or match_lower.startswith('graph'):
            if not result.flowchart:
                result.flowchart = match_stripped
        elif match_lower.startswith('sequencediagram'):
            if not result.sequence:
                result.sequence = match_stripped
        elif '->>' in match or '->>>' in match:
            if not result.sequence:
                result.sequence = match_stripped

    if result.has_any():
        return result

    _extract_mermaid_from_text(content, result)
    if result.has_any():
        return result

    for pattern_str, attr in [
        (r'`(flowchart[\s\S]*?)`', 'flowchart'),
        (r'`(graph\s+\w+[\s\S]*?)`', 'flowchart'),
        (r'`(sequenceDiagram[\s\S]*?)`', 'sequence'),
    ]:
        m = re.search(pattern_str, content, re.IGNORECASE)
        if m:
            setattr(result, attr, m.group(1).strip())
            return result

    bare_fc = re.compile(
        r'^((?:flowchart|graph)\s+\w+.*(?:\n(?:    |\t| {2,}).*)*)',
        re.MULTILINE | re.IGNORECASE,
    )
    bare_match = bare_fc.search(content)
    if bare_match:
        candidate = bare_match.group(1).strip()
        if len(candidate) > 30:
            result.flowchart = candidate

    return result


def _extract_mermaid_from_text(content: str, result: 'MermaidCodes') -> None:
    """从纯文本中按行扫描提取 Mermaid 代码"""
    current_type = None  # 'flowchart' 或 'sequence'
    current_lines: List[str] = []

    def _flush():
        nonlocal current_type, current_lines
        if current_type and current_lines:
            code = '\n'.join(current_lines)
            if current_type == 'flowchart' and not result.flowchart:
                result.flowchart = code
            elif current_type == 'sequence' and not result.sequence:
                result.sequence = code
        current_type = None
        current_lines = []

    for line in content.split('\n'):
        ls = line.strip()
        ll = ls.lower()
        if ll.startswith('flowchart') or ll.startswith('graph '):
            _flush()
            current_type = 'flowchart'
            current_lines = [line]
        elif ll.startswith('sequencediagram'):
            _flush()
            current_type = 'sequence'
            current_lines = [line]
        elif current_type:
            if ls.startswith('分析') or ls.startswith('总结'):
                _flush()
            elif ls and not ls.startswith('#'):
                current_lines.append(line)
    _flush()


def _build_knowledge_graph_from_json(
    graph_json: dict,
    source_tg: str = "",
    tg_title: str = "",
    parse_method: str = "llm",
) -> KnowledgeGraph:
    """将 entities/relations 或 nodes/edges JSON 转换为 KnowledgeGraph"""
    graph = KnowledgeGraph()
    graph.metadata = {
        "source_tg": source_tg,
        "tg_title": tg_title,
        "parse_method": parse_method,
    }

    id_to_name: Dict[str, str] = {}

    for entity in graph_json.get("entities", graph_json.get("nodes", [])):
        if not isinstance(entity, dict):
            continue
        entity_id = entity.get("id", entity.get("name", ""))
        name = entity.get("name", entity.get("text", entity_id))
        if not name:
            continue

        node_type = NodeType.CONDITION if entity.get("type") == "condition" else NodeType.EVENT
        id_to_name[entity_id] = name

        graph.add_node(GraphNode(
            name=name,
            node_type=node_type,
            properties=entity.get("properties", entity),
            source_tg=[source_tg] if source_tg else [],
        ))

    for rel in graph_json.get("relations", graph_json.get("edges", [])):
        if not isinstance(rel, dict):
            continue
        src_key = rel.get("source", rel.get("source_id", rel.get("from", "")))
        tgt_key = rel.get("target", rel.get("target_id", rel.get("to", "")))
        src = id_to_name.get(src_key)
        tgt = id_to_name.get(tgt_key)
        if not src or not tgt:
            continue

        graph.add_edge(GraphEdge(
            source_id=src,
            target_id=tgt,
            edge_type=EdgeType.CAUSAL,
            relation_name=rel.get("relation", rel.get("relation_name", rel.get("label", "导致"))),
            source_tg=source_tg,
            properties=rel.get("properties", {}),
        ))

    return graph


@dataclass
class MermaidNode:
    """Mermaid 节点"""
    id: str
    text: str
    node_type: str  # "event"（方括号）或 "condition"（花括号）


@dataclass
class MermaidEdge:
    """Mermaid 边"""
    source_id: str
    target_id: str
    label: str = ""
    edge_style: str = "-->"  # --> 实线, -.-> 虚线


@dataclass
class SequenceMessage:
    """时序图消息"""
    sender: str           # 发送方
    receiver: str         # 接收方
    message: str          # 消息内容
    message_type: str     # 消息类型: sync(->>) async(-->>)
    order: int = 0        # 消息顺序


class MermaidParser:
    """Mermaid 代码解析器，支持 flowchart 和 sequenceDiagram"""
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "",
        use_llm_parse: bool = True,
        token_stats=None,
    ):
        """初始化 Mermaid 解析器"""
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.use_llm_parse = use_llm_parse
        self.token_stats = token_stats
        self._client = None
        
        # 节点模式: ID["文本"] / ID{"文本"} / ID{{"文本"}} 等
        self.node_pattern = re.compile(
            r'([A-Za-z_][A-Za-z0-9_]*)\s*'
            r'('
            r'\[\"([^\"]*)\"\]|'
            r'\[([^\]]*)\]|'
            r'\{\{\"([^\"]*)\"\}\}|'
            r'\{\{([^\}]*)\}\}|'
            r'\{\"([^\"]*)\"\}|'
            r'\{([^\}]*)\}'
            r')',
            re.UNICODE
        )
        
        # 边模式: --> / -.-> / --label--> / |label| 等
        self.edge_pattern = re.compile(
            r'([A-Za-z_][A-Za-z0-9_]*)\s*'
            r'(?:'
            r'--\s*([^->\s][^->]*?)\s*-->|'
            r'-\.\s*([^.>\s][^.>]*?)\s*\.->|'
            r'(-->|-.->)\s*(?:\|([^|]*)\|)?'
            r')\s*'
            r'([A-Za-z_][A-Za-z0-9_]*)',
            re.UNICODE
        )
        
        self.chain_pattern = re.compile(
            r'([A-Za-z_][A-Za-z0-9_]*(?:\s*(?:-->|-.->)\s*[A-Za-z_][A-Za-z0-9_]*)+)',
            re.UNICODE
        )
        
        self.participant_pattern = re.compile(
            r'participant\s+(\w+)(?:\s+as\s+(.+?))?$',
            re.IGNORECASE | re.MULTILINE
        )
        
        self.message_pattern = re.compile(
            r'(\w+)\s*(->>|-->>|->|-->)\s*(\w+)\s*:\s*(.+?)$',
            re.MULTILINE
        )
    
    def _get_client(self):
        """获取 OpenAI 客户端，延迟初始化"""
        if self._client is None:
            from src.utils.openai_client import create_openai_client
            self._client = create_openai_client(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    def parse(self, mermaid_code: str, chinese_only: bool = False) -> Tuple[List[MermaidNode], List[MermaidEdge]]:
        """解析 Mermaid flowchart 代码，返回 (节点列表, 边列表)"""
        nodes: Dict[str, MermaidNode] = {}
        edges: List[MermaidEdge] = []

        content_lines = []
        for line in mermaid_code.strip().split('\n'):
            line = line.strip()
            if line.startswith('flowchart') or line.startswith('graph'):
                continue
            if line.startswith('%%'):
                continue
            if line:
                content_lines.append(line)
        content = '\n'.join(content_lines)

        id_to_text: Dict[str, str] = {}
        for match in self.node_pattern.finditer(content):
            node_id = match.group(1)
            text = (match.group(3) or match.group(4) or match.group(5) or
                    match.group(6) or match.group(7) or match.group(8) or "").strip()

            if chinese_only:
                if not text or not self._contains_chinese(text):
                    continue
                id_to_text[node_id] = text

            node_type = "condition" if '{' in match.group(0) else "event"
            nodes[node_id] = MermaidNode(id=node_id, text=text, node_type=node_type)

        # 移除节点定义文本，只保留 ID 以便边正则匹配
        edge_content = re.sub(r'\{\{\"[^\"]*\"\}\}', '', content)
        edge_content = re.sub(r'\{\{[^\}]*\}\}', '', edge_content)
        edge_content = re.sub(r'\{\"[^\"]*\"\}', '', edge_content)
        edge_content = re.sub(r'\{[^\}]*\}', '', edge_content)
        edge_content = re.sub(r'\[\"[^\"]*\"\]', '', edge_content)
        edge_content = re.sub(r'\[[^\]]*\]', '', edge_content)

        merge_pattern = re.compile(
            r'([A-Za-z_][A-Za-z0-9_]*(?:\s*&\s*[A-Za-z_][A-Za-z0-9_]*)+)\s*'
            r'(-->|-.->)\s*(?:\|([^|]*)\|)?\s*'
            r'([A-Za-z_][A-Za-z0-9_]*)',
            re.UNICODE,
        )

        processed_edges: set = set()

        def _should_add_edge(src: str, tgt: str) -> bool:
            if not chinese_only:
                return True
            return src in id_to_text and tgt in id_to_text

        def _ensure_default_node(nid: str):
            if not chinese_only and nid not in nodes:
                nodes[nid] = MermaidNode(id=nid, text=nid, node_type="event")

        for match in merge_pattern.finditer(edge_content):
            sources_str = match.group(1)
            arrow_type = match.group(2)
            label = (match.group(3) or "").strip()
            target_id = match.group(4)

            for source_id in (s.strip() for s in sources_str.split('&')):
                if not source_id or not _should_add_edge(source_id, target_id):
                    continue
                edge_key = (source_id, target_id, label)
                if edge_key not in processed_edges:
                    edges.append(MermaidEdge(source_id=source_id, target_id=target_id,
                                            label=label, edge_style=arrow_type))
                    processed_edges.add(edge_key)
                _ensure_default_node(source_id)
            _ensure_default_node(target_id)

        for match in self.edge_pattern.finditer(edge_content):
            source_id = match.group(1)
            target_id = match.group(6)

            # 跳过 "A & B --> C" 合并语法中已处理的部分
            ms = match.start()
            if ms > 0:
                cp = ms - 1
                while cp >= 0 and edge_content[cp] in ' \t':
                    cp -= 1
                if cp >= 0 and edge_content[cp] == '&':
                    continue

            if match.group(2):
                label, edge_style = match.group(2).strip(), "-->"
            elif match.group(3):
                label, edge_style = match.group(3).strip(), "-.->"
            else:
                edge_style = match.group(4) or "-->"
                label = (match.group(5) or "").strip()

            if not _should_add_edge(source_id, target_id):
                continue

            edge_key = (source_id, target_id, label)
            if edge_key not in processed_edges:
                edges.append(MermaidEdge(source_id=source_id, target_id=target_id,
                                        label=label, edge_style=edge_style))
                processed_edges.add(edge_key)
            _ensure_default_node(source_id)
            _ensure_default_node(target_id)

        return list(nodes.values()), edges

    def parse_chinese_only(self, mermaid_code: str) -> Tuple[List[MermaidNode], List[MermaidEdge]]:
        """解析 Mermaid 代码，只保留含中文内容的节点和边"""
        return self.parse(mermaid_code, chinese_only=True)
    
    def _contains_chinese(self, text: str) -> bool:
        """检查文本是否包含中文"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    
    def parse_sequence_diagram(self, mermaid_code: str) -> Tuple[Dict[str, str], List[SequenceMessage]]:
        """解析时序图代码，返回 (参与者字典, 消息列表)"""
        participants = {}  # ID → 别名
        messages = []
        
        lines = mermaid_code.strip().split('\n')
        message_order = 0
        
        for line in lines:
            line = line.strip()
            
            if not line or line.lower().startswith('sequencediagram'):
                continue
            if line.startswith('%%'):
                continue
            if line.lower() in ('alt', 'else', 'end', 'loop', 'opt', 'par', 'and'):
                continue
            
            participant_match = self.participant_pattern.match(line)
            if participant_match:
                p_id = participant_match.group(1)
                p_alias = participant_match.group(2) or p_id
                participants[p_id] = p_alias.strip()
                continue
            
            message_match = self.message_pattern.match(line)
            if message_match:
                sender = message_match.group(1)
                arrow_type = message_match.group(2)
                receiver = message_match.group(3)
                message_text = message_match.group(4).strip()
                
                if arrow_type in ('->>>', '->>'):
                    msg_type = 'sync'
                else:
                    msg_type = 'async'
                
                if sender not in participants:
                    participants[sender] = sender
                if receiver not in participants:
                    participants[receiver] = receiver
                
                messages.append(SequenceMessage(
                    sender=sender,
                    receiver=receiver,
                    message=message_text,
                    message_type=msg_type,
                    order=message_order,
                ))
                message_order += 1
        
        return participants, messages
    
    def to_knowledge_graph(
        self,
        mermaid_code: str,
        source_tg: str = "",
        tg_title: str = "",
    ) -> KnowledgeGraph:
        """用 LLM 将 Mermaid 代码转换为 KnowledgeGraph"""
        return self._llm_to_graph(
            system_prompt=self._build_llm_parse_prompt(),
            user_content=f"请解析以下 Mermaid 代码：\n\n```mermaid\n{mermaid_code}\n```",
            source_tg=source_tg,
            tg_title=tg_title,
            caller="LLM 解析",
        )
    
    def _llm_to_graph(
        self,
        system_prompt: str,
        user_content: str,
        source_tg: str = "",
        tg_title: str = "",
        caller: str = "LLM",
        max_retries: Optional[int] = None,
    ) -> KnowledgeGraph:
        """调用 LLM 解析内容并构建 KnowledgeGraph"""
        import time
        from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS, LOG_VERBOSE
        _max = API_RETRY_MAX_ATTEMPTS if max_retries is None else max_retries
        _interval = API_RETRY_INTERVAL_SECONDS

        if not self.api_key or not self.model_name:
            print(f"      [警告] 未配置 API，无法使用大模型解析")
            return KnowledgeGraph()

        client = self._get_client()

        for attempt in range(_max + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                    max_tokens=4096,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False},
                )

                if self.token_stats and hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    self.token_stats.add_graph_generation(total_tokens, prompt_tokens, completion_tokens)

                content = response.choices[0].message.content or ""
                graph_json = _extract_json_robust(content)

                if graph_json:
                    return _build_knowledge_graph_from_json(graph_json, source_tg, tg_title, "llm")

                if LOG_VERBOSE:
                    print(f"      [调试] 模型返回内容: {content[:300]}...")
                raise ValueError("未能从响应中提取有效的 JSON")

            except Exception as e:
                if attempt < _max:
                    print(f"      [重试] 第{attempt+1}次失败，{_interval}秒后重试: {e}")
                    time.sleep(_interval)
                else:
                    print(f"      [错误] {caller}失败: {e}")
                    return KnowledgeGraph()

        return KnowledgeGraph()

    def mermaid_codes_to_knowledge_graph(
        self,
        mermaid_codes: 'MermaidCodes',
        source_tg: str = "",
        tg_title: str = "",
        keywords: Optional[List[str]] = None,
    ) -> KnowledgeGraph:
        """将 MermaidCodes 转换为 KnowledgeGraph（按 use_llm_parse 选择 LLM 或规则解析）"""
        if not mermaid_codes.has_any():
            return KnowledgeGraph()
        
        if self.use_llm_parse:
            graph = self._mermaid_codes_to_graph_with_llm(mermaid_codes, source_tg, tg_title)
        else:
            graph = self._mermaid_codes_to_graph_with_rules(mermaid_codes, source_tg, tg_title)
        
        if keywords:
            for node in graph.nodes.values():
                node.properties["keywords"] = keywords
        
        return graph
    
    def _mermaid_codes_to_graph_with_llm(
        self,
        mermaid_codes: 'MermaidCodes',
        source_tg: str = "",
        tg_title: str = "",
    ) -> KnowledgeGraph:
        """用 LLM 将 MermaidCodes 转换为 KnowledgeGraph"""
        if mermaid_codes.flowchart and not mermaid_codes.sequence:
            return self.to_knowledge_graph(mermaid_codes.flowchart, source_tg, tg_title)

        if mermaid_codes.sequence and not mermaid_codes.flowchart:
            return self.to_knowledge_graph(mermaid_codes.sequence, source_tg, tg_title)

        combined = self._combine_mermaid_codes(mermaid_codes)
        return self._llm_to_graph(
            system_prompt=self._build_combined_parse_prompt(),
            user_content=f"请解析以下 Mermaid 代码（包含流程图和时序图）：\n\n{combined}",
            source_tg=source_tg,
            tg_title=tg_title,
            caller="LLM 联合解析",
        )
    
    def _mermaid_codes_to_graph_with_rules(
        self,
        mermaid_codes: 'MermaidCodes',
        source_tg: str = "",
        tg_title: str = "",
    ) -> KnowledgeGraph:
        """用正则规则将 MermaidCodes 转换为 KnowledgeGraph"""
        graph = KnowledgeGraph()
        graph.metadata = {
            "source_tg": source_tg,
            "tg_title": tg_title,
            "parse_method": "rules",
        }
        
        if mermaid_codes.flowchart:
            self._parse_flowchart_with_rules(mermaid_codes.flowchart, source_tg, graph)
        
        if mermaid_codes.sequence:
            self._parse_sequence_with_rules(mermaid_codes.sequence, source_tg, graph)
        
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"      [规则解析] 成功提取 {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
        return graph
    
    def _parse_flowchart_with_rules(
        self,
        mermaid_code: str,
        source_tg: str,
        graph: KnowledgeGraph,
    ) -> None:
        """规则解析 flowchart 代码并添加到图谱"""
        nodes, edges = self.parse_chinese_only(mermaid_code)
        for mermaid_node in nodes:
            node_type = NodeType.CONDITION if mermaid_node.node_type == "condition" else NodeType.EVENT
            
            graph_node = GraphNode(
                name=mermaid_node.text,
                node_type=node_type,
                properties={
                    "mermaid_id": mermaid_node.id,
                    "mermaid_type": mermaid_node.node_type,
                },
                source_tg=[source_tg] if source_tg else [],
            )
            graph.add_node(graph_node)
        
        id_to_name = {node.id: (node.text or node.id) for node in nodes}
        
        for mermaid_edge in edges:
            source_name = id_to_name.get(mermaid_edge.source_id)
            target_name = id_to_name.get(mermaid_edge.target_id)
            
            if not source_name or not target_name:
                continue
            
            # 虚线 -> 触发，实线 -> 因果
            if mermaid_edge.edge_style == "-.->":
                edge_type = EdgeType.TRIGGERS
            else:
                edge_type = EdgeType.CAUSAL
            
            relation_name = mermaid_edge.label if mermaid_edge.label else ""
            
            graph_edge = GraphEdge(
                source_id=source_name,
                target_id=target_name,
                edge_type=edge_type,
                relation_name=relation_name,
                source_tg=source_tg,
                properties={
                    "mermaid_style": mermaid_edge.edge_style,
                },
            )
            graph.add_edge(graph_edge)
    
    def _parse_sequence_with_rules(
        self,
        mermaid_code: str,
        source_tg: str,
        graph: KnowledgeGraph,
    ) -> None:
        """规则解析 sequenceDiagram 代码并添加到图谱"""
        participants, messages = self.parse_sequence_diagram(mermaid_code)
        
        prev_node_name = None
        for msg in messages:
            sender_name = participants.get(msg.sender, msg.sender)
            receiver_name = participants.get(msg.receiver, msg.receiver)
            node_name = f"{sender_name} -> {receiver_name}: {msg.message}"
            
            if node_name in graph.nodes:
                prev_node_name = node_name
                continue
            
            graph_node = GraphNode(
                name=node_name,
                node_type=NodeType.EVENT,
                properties={
                    "sender": sender_name,
                    "receiver": receiver_name,
                    "message": msg.message,
                    "message_type": msg.message_type,
                    "order": msg.order,
                },
                source_tg=[source_tg] if source_tg else [],
            )
            graph.add_node(graph_node)
            
            if prev_node_name and prev_node_name != node_name:
                graph_edge = GraphEdge(
                    source_id=prev_node_name,
                    target_id=node_name,
                    edge_type=EdgeType.CAUSAL,
                    source_tg=source_tg,
                )
                graph.add_edge(graph_edge)
            
            prev_node_name = node_name
    
    def _combine_mermaid_codes(self, mermaid_codes: 'MermaidCodes') -> str:
        """将流程图和时序图拼接为一段输入文本"""
        parts = []
        
        if mermaid_codes.flowchart:
            parts.append("## 流程图 (Flowchart)\n```mermaid\n" + mermaid_codes.flowchart + "\n```")
        
        if mermaid_codes.sequence:
            parts.append("## 时序图 (Sequence Diagram)\n```mermaid\n" + mermaid_codes.sequence + "\n```")
        
        return "\n\n".join(parts)
    
    
    def _build_combined_parse_prompt(self) -> str:
        """构建联合解析流程图+时序图的提示词"""
        return '''# Role
你是一位专业的工业知识图谱构建专家，擅长从 Mermaid 图表代码中提取结构化知识。

# Task
将 **流程图和时序图** 一起解析为一个统一的结构化 JSON 格式。

**重要**：流程图和时序图描述的是同一个流程/场景，它们之间可能存在重复或关联的内容。
你需要：
1. 识别两个图中描述相同事件/操作的内容，合并为同一个实体
2. 综合两个图的信息，提取更完整的实体属性
3. 输出一个去重、统一的图谱

# 输出格式

请严格按照以下 JSON 格式输出：

```json
{
  "entities": [
    {
      "id": "e1",
      "name": "实体/事件的完整描述",
      "type": "operation|event|state|condition",
      "entity": "涉及的设备/对象名称",
      "parameter": "相关参数名称",
      "state": "状态值或条件",
      "source": "flowchart|sequence|both"
    }
  ],
  "relations": [
    {
      "source": "e1",
      "target": "e2", 
      "relation": "关系类型",
      "properties": {}
    }
  ]
}
```

# 字段说明

## entities 实体字段：
- **id**: 实体唯一标识，使用 e1, e2, e3... 的格式
- **name**: 实体的完整描述
- **type**: 实体类型
  - `operation`: 操作动作
  - `event`: 系统事件
  - `state`: 状态描述
  - `condition`: 条件判断
- **entity**: 涉及的具体设备或对象
- **parameter**: 相关的参数
- **state**: 具体的状态值
- **source**: 来源标记
  - `flowchart`: 仅来自流程图
  - `sequence`: 仅来自时序图
  - `both`: 两个图中都出现（已合并）

## relations 关系字段：
- **source**: 源实体 ID
- **target**: 目标实体 ID
- **relation**: 关系类型（后续步骤、触发、导致、检测判断、条件分支等）
- **properties**: 附加属性

# 合并规则

1. **相同事件识别**：如果流程图和时序图中描述的是同一个操作/事件（即使措辞略有不同），应合并为一个实体
2. **信息补充**：从两个图中提取最完整的信息填充实体属性
3. **关系整合**：综合两个图的边/消息，建立完整的关系网络
4. **去重**：确保最终输出不包含重复的实体或关系

# 重要提示
1. 必须输出有效的 JSON 格式
2. 实体 ID 必须与关系中的 source/target 对应
3. 优先使用更具体、更完整的描述作为实体名称
'''
    
    def _build_llm_parse_prompt(self) -> str:
        """构建 LLM 解析 Mermaid 的提示词"""
        return '''# Role
你是一位专业的工业知识图谱构建专家，擅长从 Mermaid 图表代码中提取结构化知识。

# Task
将 Mermaid 代码（流程图或时序图）解析为结构化的 JSON 格式，提取其中的实体和关系。

# 输出格式

请严格按照以下 JSON 格式输出：

```json
{
  "entities": [
    {
      "id": "e1",
      "name": "实体/事件的完整描述",
      "type": "operation|event|state|condition",
      "entity": "涉及的设备/对象名称",
      "parameter": "相关参数名称",
      "state": "状态值或条件"
    }
  ],
  "relations": [
    {
      "source": "e1",
      "target": "e2", 
      "relation": "关系类型",
      "properties": {}
    }
  ]
}
```

# 字段说明

## entities 实体字段：
- **id**: 实体唯一标识，使用 e1, e2, e3... 的格式
- **name**: 实体的完整描述（直接使用 Mermaid 节点中的文本）
- **type**: 实体类型
  - `operation`: 操作动作（如"按下按钮"、"打开阀门"）
  - `event`: 系统事件（如"系统启动"、"报警触发"）
  - `state`: 状态描述（如"温度正常"、"压力过高"）
  - `condition`: 条件判断（如"是否达标?"、"油压是否建立?"）
- **entity**: 涉及的具体设备或对象（如"油泵P-101"、"控制系统"）
- **parameter**: 相关的参数（如"液位"、"压力"、"温度"）
- **state**: 具体的状态值（如">500mm"、"开启"、"报错"）

## relations 关系字段：
- **source**: 源实体 ID
- **target**: 目标实体 ID
- **relation**: 关系类型，常见的有：
  - `后续步骤`: 按顺序执行
  - `触发`: 一个事件触发另一个
  - `导致`: 因果关系
  - `检测判断`: 进行条件检测
  - `条件分支`: 根据条件分流
- **properties**: 附加属性（如 timeout、condition 等）

# 解析规则

1. **流程图节点**：方括号 `["文本"]` 通常是 operation/event，花括号 `{{"文本?"}}` 是 condition
2. **时序图消息**：每条消息转换为一个 event 实体
3. **边/箭头**：转换为 relations，标签作为 relation 类型
4. **保持原文**：name 字段保持 Mermaid 中的原始文本

# 重要提示
1. 必须输出有效的 JSON 格式
2. 实体 ID 必须与关系中的 source/target 对应
3. 尽可能从文本中提取 entity、parameter、state 信息
4. 如果无法确定某个字段，可以留空字符串
'''
    


@dataclass
class MermaidCodes:
    """Mermaid 代码集合"""
    flowchart: str = ""
    sequence: str = ""
    
    def has_any(self) -> bool:
        """是否包含任一类型的代码"""
        return bool(self.flowchart or self.sequence)
    


@dataclass
class MermaidOnlyResult:
    """Mermaid 生成结果（两阶段模式第一阶段）"""
    mermaid_codes: MermaidCodes = field(default_factory=MermaidCodes)
    keywords: List[str] = field(default_factory=list)
    raw_response: str = ""
    success: bool = False
    error_message: str = ""
    tg_id: str = ""
    


class MermaidOnlyGenerator:
    """两阶段模式第一阶段：从文档生成 Mermaid 代码，第二阶段再解析为图谱"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        output_dir: Optional[str] = None,
        token_stats=None,
        max_concurrent: int = 16,
    ):
        """初始化 MermaidOnlyGenerator"""
        from src.utils.openai_client import create_openai_client, APIRateLimiter
        
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.token_stats = token_stats
        self.rate_limiter = APIRateLimiter(max_concurrent=max_concurrent, base_cooldown=5.0)
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_valid_mermaid(self, code: str) -> bool:
        """验证是否为有效的 Mermaid 代码（非占位符，含关键字和图结构）"""
        if not code or not isinstance(code, str):
            return False
        
        code_stripped = code.strip().lower()
        
        invalid_patterns = [
            '...',      # 省略号
            '…',        # 省略号（单字符）
            '......',   # 多个点
            '[待补充]',
            '[placeholder]',
            'todo',
            'tbd',
        ]
        if code_stripped in invalid_patterns or code_stripped.replace('.', '') == '':
            return False
        
        if len(code_stripped) < 20:
            return False
        
        valid_starts = ['flowchart', 'graph', 'sequencediagram', 'classDiagram', 'statediagram', 'erdiagram', 'journey', 'gantt', 'pie', 'mindmap']
        if not any(code_stripped.startswith(start.lower()) for start in valid_starts):
            return False
        
        has_structure = (
            '-->' in code or
            '-.>' in code or
            '==>' in code or
            '->>' in code or
            '-->>>' in code or
            '[' in code or
            '{' in code
        )
        
        return has_structure
    
    def _build_prompt(self) -> str:
        """构建 Mermaid 生成的系统提示词"""
        
        prompt = '''
        # Role
你是工业流程文档分析专家，擅长从SOP文档中**完整、精确**地提取流程结构，生成全中文标签的Mermaid流程图，并提取文档的核心产品，产品名称要完整。

# 核心任务
1. 提取文档的**核心关键词**（2-3个，聚焦于产品品牌、名称、型号及关键故障、部位和操作），一般在文档的开头
2. 从工业SOP文档中提取**完整的流程结构**，生成**节点标签和边标签全中文**的Mermaid流程图。

# 全中文输出要求（极其重要）

## 严格禁止
1. **禁止简化**：不要把多个条件合并成一个
2. **禁止编造**：只提取文档中明确描述的内容
3. **禁止循环**：不生成"A→B"和"B→A"的回环结构
4. **禁止步骤编号**：节点标签中不包含"5.1"、"步骤3"等编号

## 语言规范
1. **节点ID**：使用英文命名（如 `check_temp`、`replace_seal`）
2. **节点标签**：必须全中文（方括号和花括号内的文字）
4. **禁止英文标签**：节点标签和边标签中不允许出现英文

## 完整性要求
1. **完整提取**：文档中提到的每一个操作步骤、每一个判断条件、每一个分支都必须体现
2. **保留阈值**：所有数值条件必须完整保留（如"温度≤65℃"、"振动>2.8mm/s"）
3. **多分支判断**：当文档描述多个条件分支时，必须全部体现，文档不存在的分支不要编造

# 节点格式规范

## 正确示例（ID英文，标签中文）
```mermaid
flowchart TD
    start[开始设备维护] --> safety_check{{是否完成能量隔离?}}
    safety_check -->|已隔离| isolation_ok[隔离确认完成]
    safety_check -->|未隔离| do_loto[执行上锁挂牌程序]
    do_loto --> safety_check
    
    isolation_ok --> visual_inspect[外观与泄漏检查]
    visual_inspect --> temp_measure[测量轴承温度]
    
    temp_measure --> temp_judge{{温度判断}}
    temp_judge -->|温度≤65℃| temp_ok[温度正常，记录数据]
    temp_judge -->|65℃<温度≤75℃| check_lube[检查润滑状态]
    temp_judge -->|温度>75℃| emergency_stop[立即停机并报警]
    
    check_lube --> lube_status{{润滑状态判断}}
    lube_status -->|油脂干涸或乳化| replace_lube[更换润滑脂]
    lube_status -->|油脂正常| check_clearance[测量轴承游隙]
    lube_status -->|混入异物或水| clean_replace[清洁后更换新脂]
```

# 输出格式
```json
{
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "mermaid_code": ""
}
```

# Task
请从以下SOP文档中**完整提取**流程结构，生成Mermaid流程图。

**关键要求：**
1. 节点ID使用英文，节点标签和边标签必须全中文
2. 提取文档中**所有**操作步骤，不要遗漏
3. 保留**完整的数值条件**
4. 不要编造文档中不存在的分支条件等逻辑

文本内容：
'''
        return prompt
    
    def generate(
        self,
        text: str,
        tg_id: str = "",
        max_retries: Optional[int] = None,
    ) -> MermaidOnlyResult:
        """从文档文本生成 Mermaid 代码和关键词"""
        import time
        from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS
        from src.utils.openai_client import check_finish_reason, is_rate_limit_error
        _max = API_RETRY_MAX_ATTEMPTS if max_retries is None else max_retries
        _interval = API_RETRY_INTERVAL_SECONDS
        
        prompt = self._build_prompt()
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
        
        from src.config import LOG_VERBOSE
        
        if LOG_VERBOSE:
            print(f"\n{'='*60}")
            print(f"[LLM输入] 文档: {tg_id}, 模型: {self.model_name}")
            print(f"[LLM输入] user message 长度: {len(text)} 字符")
            print(f"{'='*60}")
        
        for attempt in range(_max + 1):
            try:
                with self.rate_limiter.acquire():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=8192,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False},
                    )
                
                self.rate_limiter.reset_429_counter()
                finish_reason = check_finish_reason(response, caller=f"Mermaid-{tg_id}")
                
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0
                if self.token_stats and hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    self.token_stats.add_graph_generation(total_tokens, prompt_tokens, completion_tokens)
                
                content = response.choices[0].message.content or ""
                
                if LOG_VERBOSE:
                    print(f"\n{'='*60}")
                    print(f"[LLM输出] 文档: {tg_id}, finish_reason={finish_reason}, 输出长度: {len(content)} 字符")
                    print(content)
                    print(f"{'='*60}")
                
                if finish_reason == "length":
                    raise ValueError(f"响应被截断 (finish_reason=length, output_tokens={completion_tokens})，需要重试")
                
                parsed_result = self._parse_json_response(content)
                keywords = parsed_result.get("keywords", [])
                mermaid_code_str = parsed_result.get("mermaid_code", "")
                
                mermaid_codes = MermaidCodes()
                if mermaid_code_str and self._is_valid_mermaid(mermaid_code_str):
                    mermaid_code_str = mermaid_code_str.replace("\\n", "\n")
                    mermaid_code_lower = mermaid_code_str.lower().strip()
                    if mermaid_code_lower.startswith("flowchart") or mermaid_code_lower.startswith("graph"):
                        mermaid_codes.flowchart = mermaid_code_str
                    elif mermaid_code_lower.startswith("sequencediagram"):
                        mermaid_codes.sequence = mermaid_code_str
                    else:
                        mermaid_codes = self._extract_mermaid(content)
                
                if not mermaid_codes.has_any():
                    mermaid_codes = self._extract_mermaid(content)
                
                if mermaid_codes.has_any():
                    if LOG_VERBOSE:
                        selected_types = []
                        if mermaid_codes.flowchart:
                            selected_types.append("流程图")
                        if mermaid_codes.sequence:
                            selected_types.append("时序图")
                        print(f"      [图表选择] 模型选择输出: {', '.join(selected_types)}")
                        if keywords:
                            print(f"      [关键词] 提取到关键词: {keywords}")
                    
                    if self.output_dir:
                        self._save_mermaid_to_file(mermaid_codes, tg_id)
                    
                    return MermaidOnlyResult(
                        mermaid_codes=mermaid_codes,
                        keywords=keywords,
                        raw_response=content,
                        success=True,
                        tg_id=tg_id,
                    )
                
                if LOG_VERBOSE:
                    print(f"      [诊断] {tg_id}: 提取失败，原始输出前300字符: {repr(content[:300])}")
                raise ValueError("未能提取到任何Mermaid代码，模型可能未按预期格式输出")
                
            except Exception as e:
                if attempt < _max:
                    if is_rate_limit_error(e):
                        self.rate_limiter.report_rate_limit()
                    print(f"      [重试] {tg_id}: 第{attempt+1}次失败，{_interval}s后重试: {e}")
                    time.sleep(_interval)
                else:
                    return MermaidOnlyResult(
                        success=False,
                        error_message=f"Mermaid生成失败: {e}",
                        tg_id=tg_id,
                    )
        
        return MermaidOnlyResult(
            success=False,
            error_message="超出最大重试次数",
            tg_id=tg_id,
        )
    
    def _parse_json_response(self, content: str) -> dict:
        """解析模型输出中的 JSON（兼容反引号包裹等非标准格式）"""
        
        result = {"keywords": [], "mermaid_code": ""}
        
        # 模型可能输出 "mermaid_code": `...`（非法 JSON 但常见）
        backtick_pattern = re.compile(
            r'"mermaid_code"\s*:\s*`([\s\S]*?)`',
        )
        backtick_match = backtick_pattern.search(content)
        if backtick_match:
            mermaid_from_backtick = backtick_match.group(1).strip()
            if mermaid_from_backtick:
                result["mermaid_code"] = mermaid_from_backtick
            
            kw_pattern = re.compile(r'"keywords"\s*:\s*\[([^\]]*)\]')
            kw_match = kw_pattern.search(content)
            if kw_match:
                try:
                    result["keywords"] = json.loads("[" + kw_match.group(1) + "]")
                except (json.JSONDecodeError, Exception):
                    pass
            
            if result["mermaid_code"]:
                return result
        
        try:
            json_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)```', re.IGNORECASE)
            json_matches = json_pattern.findall(content)
            
            for match in json_matches:
                match_stripped = match.strip()
                try:
                    parsed = json.loads(match_stripped)
                    if isinstance(parsed, dict):
                        if "keywords" in parsed:
                            result["keywords"] = parsed["keywords"]
                        if "mermaid_code" in parsed:
                            result["mermaid_code"] = parsed["mermaid_code"]
                        if result["mermaid_code"]:
                            return result
                except json.JSONDecodeError:
                    continue
            
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx + 1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        if "keywords" in parsed:
                            result["keywords"] = parsed["keywords"]
                        if "mermaid_code" in parsed:
                            result["mermaid_code"] = parsed["mermaid_code"]
                        if result["mermaid_code"]:
                            return result
                except json.JSONDecodeError:
                    pass
                
        except Exception:
            pass
        
        return result
    
    def _extract_mermaid(self, content: str) -> MermaidCodes:
        """从模型输出中提取 Mermaid 代码"""
        return _extract_mermaid_codes(content)
    
    def _save_mermaid_to_file(self, mermaid_codes: "MermaidCodes", tg_id: str) -> None:
        """将 Mermaid 代码保存到 .mmd 文件"""
        if not self.output_dir:
            return
        
        output_file = self.output_dir / f"{tg_id}_mermaid.mmd"
        
        content_parts = []
        content_parts.append(f"%% 文档ID: {tg_id}")
        content_parts.append(f"%% 生成时间: {self._get_timestamp()}")
        content_parts.append("")
        
        if mermaid_codes.flowchart:
            content_parts.append("%% ========== 流程图 (Flowchart) ==========")
            content_parts.append(mermaid_codes.flowchart)
            content_parts.append("")
        
        if mermaid_codes.sequence:
            content_parts.append("%% ========== 时序图 (Sequence Diagram) ==========")
            content_parts.append(mermaid_codes.sequence)
            content_parts.append("")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(content_parts))
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    


@dataclass
class GraphGenerationResult:
    """图谱生成结果"""
    graph: Optional[KnowledgeGraph]
    mermaid_codes: MermaidCodes = field(default_factory=MermaidCodes)
    raw_response: str = ""
    success: bool = False
    error_message: str = ""
    


class GraphGenerator:
    """调用微调模型直接生成图谱（<think> 内含 Mermaid，主输出为 JSON）"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        token_stats=None,
        max_concurrent: int = 16,
    ):
        """初始化 GraphGenerator"""
        from src.utils.openai_client import create_openai_client, APIRateLimiter
        
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.token_stats = token_stats
        self.rate_limiter = APIRateLimiter(max_concurrent=max_concurrent, base_cooldown=5.0)
        
        self.mermaid_parser = MermaidParser(token_stats=self.token_stats)
    
    def generate(
        self,
        text: str,
        tg_id: str = "",
        max_retries: Optional[int] = None,
    ) -> GraphGenerationResult:
        """调用微调模型生成图谱"""
        import time
        from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS
        from src.utils.openai_client import is_rate_limit_error
        _max = API_RETRY_MAX_ATTEMPTS if max_retries is None else max_retries
        _interval = API_RETRY_INTERVAL_SECONDS
        
        for attempt in range(_max + 1):
            try:
                prompt = f'''
# Role
你是一位精通工业系统分析、操作流程优化与知识图谱构建的专家。你的任务是从给定的工业文本（如操作规程、系统手册、事故报告）中提取结构化的知识图谱数据。

# Goal
构建一个**以“事件/状态/操作”为核心的动态图谱**。
- 图谱不仅要涵盖故障传播，还要能清晰表达**正常的操作流程**、**系统逻辑判定**以及**设备间的交互机制**。
- 节点应当描述具体的行为或状态变化（例如：“启动供油泵”或“润滑油压力正常”），而不仅仅是静态的设备名词。

# Workflow
请严格按照以下步骤进行深度分析与输出：

1. **思维链分析**:
   - **文本解构**: 识别文本中的关键时间点、操作步骤、逻辑转折。将句子拆解为“主体(Entity) - 参数(Parameter) - 状态/动作(State)”的结构。
   - **双视图建模 (Mermaid)**:
     - **视图 A: 时序交互 (Sequence Diagram)**: 必须展示事件发生的严格**时间顺序**、**设备/人员之间的交互**、**控制信号的传递**（Request/Response）。
     - **视图 B: 逻辑演化 (Flowchart)**: 必须展示业务或故障的**因果链**、**逻辑判定分支**（如：若A则B，否则C）、以及**状态的流转**。
   - **节点定义**: 规划 JSON 实体，确保每个 `name` 都是一个完整的信息单元。

2. **JSON 输出**:
   - 输出一个包含 `entities` 和 `relations` 的标准 JSON 对象。
   - **Entities (节点结构)**:
     - `name`: 完整的事件/状态名称 (e.g., "除氧器压力下降").
     - `type`: 节点类型 (e.g., "operation"操作, "event"事件, "state"状态, "logic"逻辑 等).
     - `entity`: 涉及的主体 (e.g., "除氧器").
     - `parameter`: 涉及的参数 (e.g., "压力").
     - `state`: 具体值/状态 (e.g., "下降").

   - **Relations (关系结构)**:
     - `source`: 源节点名称。
     - `target`: 目标节点名称。
     - `relation`: 关系类型 (e.g., "步骤后续", "条件触发", "导致", "包含" 等).
     - `properties`: 关系的属性.

# Output Format Example
<think>
文本分析...
关键逻辑提取...

```mermaid
sequenceDiagram
    participant A
    participant B
    A->>B: 信号交互
    ...
```

```mermaid
flowchart TD
    A[状态1] -->|条件| B{{逻辑判定}}
    B -- Yes --> C[动作]
    ...
```
</think>

```json
{{
  "entities": [...],
  "relations": [...]
}}
```

# Few-Shot Example (参考范例)
**User Input:**
<<< 请提取知识图谱。
文本：系统启动前，操作员需确认润滑油箱液位大于500mm。确认无误后，按下启动按钮。控制系统接收信号后，开启辅助油泵P-101。若油压在10秒内未建立至0.3MPa，系统将报错并禁止主机启动。

**Model Output:**
<think>
分析文本：这是一段包含操作确认、信号传递和逻辑判定的启动流程。
1. **时序维度**：人机交互（检查、按钮） -> 系统响应（开启泵） -> 延时判定（检测压力）。
2. **逻辑维度**：前置条件（液位） -> 动作（启动） -> 分支判断（压力是否建立） -> 结果（报错或继续）。

```mermaid
sequenceDiagram
    participant OP as 操作员
    participant SYS as 控制系统
    participant PUMP as 辅助油泵P-101
    participant SENSOR as 压力传感器

    OP->>OP: 确认液位>500mm
    OP->>SYS: 按下启动按钮
    SYS->>PUMP: 发送开启指令
    PUMP->>SENSOR: 建立油压
    loop 10秒检测
        SENSOR-->>SYS: 反馈压力值
    end
    alt 压力 < 0.3MPa
        SYS->>OP: 报错并禁止启动
    end
```

```mermaid
flowchart TD
    S1[操作员确认液位>500mm] -->|前置满足| S2[按下启动按钮]
    S2 -->|触发| S3[系统开启辅助油泵P-101]
    S3 --> S4{{10秒内油压>=0.3MPa?}}
    S4 -- No --> S5[系统报错并禁止主机启动]
    S4 -- Yes --> S6[准备主机启动]
```
</think>

```json
{{
  "entities": [
    {{      
      "name": "确认润滑油箱液位>500mm",
      "type": "operation",
      "entity": "润滑油箱",
      "parameter": "液位",
      "state": ">500mm"
    }},
    {{      
      "name": "按下启动按钮",
      "type": "operation",
      "entity": "启动按钮",
      "parameter": "状态",
      "state": "按下"
    }},
    {{      
      "name": "系统开启辅助油泵P-101",
      "type": "event",
      "entity": "辅助油泵P-101",
      "parameter": "运行状态",
      "state": "开启"
    }},
    {{      
      "name": "油压未建立至0.3MPa",
      "type": "state",
      "entity": "系统油压",
      "parameter": "压力",
      "state": "<0.3MPa"
    }},
    {{      
      "name": "系统报错并禁止启动",
      "type": "event",
      "entity": "控制系统",
      "parameter": "启动逻辑",
      "state": "禁止/报错"
    }},
  ],
  "relations": [
    {{ "source": "e1", "target": "e2", "relation": "后续步骤", "properties": {{}} }},
    {{ "source": "e2", "target": "e3", "relation": "触发", "properties": {{}} }},
    {{ "source": "e3", "target": "e4", "relation": "检测判断", "properties": {{"timeout": "10s"}} }},
    {{ "source": "e4", "target": "e5", "relation": "导致", "properties": {{"condition": "Timeout & Low Pressure"}} }}
  ]
}}
```

# Task
请处理以下新的输入文本
'''
                with self.rate_limiter.acquire():
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": text}
                        ],
                        temperature=0,
                        max_tokens=8192,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False},
                    )
                
                self.rate_limiter.reset_429_counter()
                
                if self.token_stats and hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    self.token_stats.add_graph_generation(total_tokens, prompt_tokens, completion_tokens)
                
                content = response.choices[0].message.content or ""
                from src.config import LOG_VERBOSE
                if LOG_VERBOSE:
                    print(f"[GraphGenerator] 文档: {tg_id}, 输出: {len(content)} 字符")
                    print(content)
                result = self._parse_response(content, tg_id)
                result.raw_response = content
                
                if result.success:
                    return result
                    
                # JSON 解析失败时回退到 Mermaid 构建图谱
                if result.mermaid_codes.flowchart:
                    if LOG_VERBOSE:
                        print(f"      [备选] JSON解析失败，尝试从flowchart Mermaid代码构建图谱...")
                    graph = self.mermaid_parser.to_knowledge_graph(
                        result.mermaid_codes.flowchart,
                        source_tg=tg_id,
                        tg_title=text.split('\n')[0] if text else tg_id,
                    )
                    if graph.nodes:
                        result.graph = graph
                        result.success = True
                        return result
                
                raise ValueError(result.error_message or "图谱生成失败")
                
            except Exception as e:
                if attempt < _max:
                    if is_rate_limit_error(e):
                        self.rate_limiter.report_rate_limit()
                    print(f"      [重试] {tg_id}: 第{attempt+1}次失败，{_interval}s后重试: {e}")
                    time.sleep(_interval)
                else:
                    return GraphGenerationResult(
                        graph=None,
                        success=False,
                        error_message=f"生成失败: {e}"
                    )
        
        return GraphGenerationResult(
            graph=None,
            success=False,
            error_message="超出最大重试次数"
        )
    
    def _parse_response(self, content: str, tg_id: str = "") -> GraphGenerationResult:
        """解析模型响应，提取 <think> 中的 Mermaid 和主输出中的图谱 JSON"""
        
        result = GraphGenerationResult(graph=None)
        
        think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        think_match = think_pattern.search(content)
        
        if think_match:
            think_content = think_match.group(1).strip()
            result.mermaid_codes = self._extract_mermaid(think_content)
        
        main_content = think_pattern.sub('', content).strip()
        graph = self._parse_graph_json(main_content, tg_id)
        
        if graph and graph.nodes:
            result.graph = graph
            result.success = True
        else:
            result.error_message = "无法从响应中解析图谱JSON"
        
        return result
    
    def _extract_mermaid(self, content: str) -> MermaidCodes:
        """从模型输出中提取 Mermaid 代码"""
        return _extract_mermaid_codes(content)
    
    def _parse_graph_json(self, content: str, tg_id: str = "") -> Optional[KnowledgeGraph]:
        """从 JSON 字符串解析图谱（兼容 nodes/edges 和 entities/relations 格式）"""
        
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        json_str = None
        for m in re.finditer(r'\{', content):
            candidate = content[m.start():]
            try:
                json.loads(candidate)
                json_str = candidate
                break
            except json.JSONDecodeError:
                continue
        
        if json_str is None:
            brace_pattern = re.compile(r'\{[\s\S]*\}')
            brace_match = brace_pattern.search(content)
            if not brace_match:
                return None
            json_str = brace_match.group()
        
        json_str = self._fix_json_errors(json_str)
        
        try:
            data = json.loads(json_str)
            
            graph = KnowledgeGraph()
            graph.metadata = {"source_tg": tg_id}
            
            nodes_data = data.get("nodes", data.get("entities", []))
            id_to_name = {}
            
            for node_data in nodes_data:
                if isinstance(node_data, dict):
                    original_id = node_data.get("id", node_data.get("name", ""))
                    node_name = node_data.get("name", node_data.get("text", original_id))
                    node_type_str = node_data.get("type", node_data.get("node_type", "event"))
                    
                    node_id = original_id
                    id_to_name[original_id] = node_id
                    node_type = self._parse_node_type(node_type_str)
                    
                    graph_node = GraphNode(
                        name=node_name,
                        node_type=node_type,
                        properties=node_data.get("properties", {}),
                        source_tg=[tg_id] if tg_id else [],
                    )
                    graph.add_node(graph_node)
            
            edges_data = data.get("edges", data.get("relations", []))
            
            for edge_data in edges_data:
                if isinstance(edge_data, dict):
                    original_source = edge_data.get("source", edge_data.get("source_id", edge_data.get("from", "")))
                    original_target = edge_data.get("target", edge_data.get("target_id", edge_data.get("to", "")))
                    relation = edge_data.get("relation", edge_data.get("relation_name", edge_data.get("label", "导致")))
                    edge_type_str = edge_data.get("type", edge_data.get("edge_type", "causal"))
                    
                    if original_source not in id_to_name or original_target not in id_to_name:
                        continue
                    
                    source_id = id_to_name[original_source]
                    target_id = id_to_name[original_target]
                    edge_type = self._parse_edge_type(edge_type_str, relation)
                    
                    graph_edge = GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                        relation_name=relation,
                        source_tg=tg_id,
                    )
                    graph.add_edge(graph_edge)
            
            return graph
            
        except json.JSONDecodeError as e:
            print(f"      [警告] JSON解析失败: {e}")
            return None
    
    def _fix_json_errors(self, json_str: str) -> str:
        """修复常见 JSON 格式错误（末尾逗号、杂乱字符等）"""
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        json_str = json_str.rstrip("'\"` \n\r\t")
        
        if not json_str.rstrip().endswith('}'):
            last_brace = json_str.rfind('}')
            if last_brace > 0:
                json_str = json_str[:last_brace + 1]
        
        return json_str
    
    def _parse_node_type(self, type_str: str) -> NodeType:
        """解析节点类型（兼容中英文）"""
        type_str = type_str.lower().strip()
        
        chinese_mapping = {
            "设备": NodeType.EQUIPMENT,
            "人员": NodeType.EVENT,      # 人员操作视为事件
            "物质": NodeType.EVENT,      # 物质状态变化视为事件
            "事件": NodeType.EVENT,
            "条件": NodeType.CONDITION,
            "判断": NodeType.CONDITION,
            "参数": NodeType.PARAMETER,
            "约束": NodeType.CONSTRAINT,
        }
        
        if type_str in chinese_mapping:
            return chinese_mapping[type_str]
        
        try:
            return NodeType(type_str)
        except ValueError:
            return NodeType.EVENT
    
    def _parse_edge_type(self, type_str: str, relation: str = "") -> EdgeType:
        """解析边类型（兼容中英文关系名）"""
        relation_lower = relation.lower() if relation else ""
        
        causal_relations = ["导致", "引发", "造成", "引起", "产生", "影响"]
        trigger_relations = ["触发", "启动", "激活", "开始"]
        
        for r in causal_relations:
            if r in relation_lower:
                return EdgeType.CAUSAL
        
        for r in trigger_relations:
            if r in relation_lower:
                return EdgeType.TRIGGERS
        
        type_lower = type_str.lower().strip() if type_str else ""
        try:
            return EdgeType(type_lower)
        except ValueError:
            return EdgeType.CAUSAL

