# -*- coding: utf-8 -*-
"""
检索阶段流水线

整合实体检索、图谱多跳检索和向量检索，提供统一的检索与答案生成接口。

Author: CongCongTian
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re as _re
import sys
import time

import numpy as np

from src.models.schemas import HybridResult, RetrievalResult, NodeType, ReasoningPath, ReasoningResult
from src.embedding.embedder import Embedder
from src.vector_store.vector_db import VectorDB
from src.graph.graph_store import GraphStore
from src.retriever.graph_retriever import GraphRetriever
from src.retriever.vector_retriever import VectorRetriever
from src.knowledge.keyword_index import KeywordIndex
from src.knowledge.keyword_extractor_local import LocalKeywordExtractor


def _safe_print(*args, verbose: bool = False, **kwargs):
    """Windows 安全的 print，自动处理 GBK 无法表示的 Unicode 字符"""
    if verbose:
        from src.config import LOG_VERBOSE
        if not LOG_VERBOSE:
            return
    try:
        print(*args, **kwargs)
    except (OSError, UnicodeEncodeError):
        text = " ".join(str(a) for a in args)
        safe_text = text.encode(sys.stdout.encoding or 'gbk', errors='replace').decode(sys.stdout.encoding or 'gbk', errors='replace')
        try:
            print(safe_text, **kwargs)
        except Exception:
            pass


def _safe_get_content(response) -> str:
    """从 LLM response 提取 content，choices 为空时返回空字符串"""
    if not response or not getattr(response, 'choices', None):
        return ""
    if len(response.choices) == 0:
        return ""
    msg = response.choices[0].message
    if not msg:
        return ""
    return msg.content or ""


class EmptyResponseError(Exception):
    """LLM 返回空响应"""
    pass


class RetrievalPipeline:
    """检索阶段流水线

    检索路径: 实体检索(关键词->陈述性知识) + 图谱检索(向量->节点->多跳) -> RRF 融合
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        persist_directory: str = "./data/flowrag_store",
        fusion_strategy: str = "rrf",
        graph_weight: float = 0.6,
        entity_weight: float = 0.4,
        expand_hops: int = 1,
        include_causal_chain: bool = True,
        max_reasoning_iterations: int = 10,
        context_token_limit: int = None,
        top_anchor_nodes: int = None,
        max_enrichment_chunks: int = None,
    ):
        from src.config import (
            RETRIEVE_LLM_API_KEY, RETRIEVE_LLM_BASE_URL, RETRIEVE_LLM_MODEL,
            RETRIEVE_EMB_API_KEY, RETRIEVE_EMB_BASE_URL, RETRIEVE_EMB_MODEL,
            CONTEXT_TOKEN_LIMIT as _CFG_CTX_LIMIT,
            TOP_ANCHOR_NODES as _CFG_TOP_ANCHOR_NODES,
            MAX_ENRICHMENT_CHUNKS as _CFG_MAX_ENRICHMENT_CHUNKS,
        )
        
        api_key = api_key or RETRIEVE_LLM_API_KEY
        base_url = base_url or RETRIEVE_LLM_BASE_URL
        embedding_model = embedding_model or RETRIEVE_EMB_MODEL
        reasoning_model = reasoning_model or RETRIEVE_LLM_MODEL
        emb_api_key = RETRIEVE_EMB_API_KEY or api_key
        emb_base_url = RETRIEVE_EMB_BASE_URL or base_url
        
        self.persist_directory = Path(persist_directory)
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_model = reasoning_model
        
        self.graph_weight = graph_weight
        self.entity_weight = entity_weight
        self.max_reasoning_iterations = max_reasoning_iterations
        self.max_enrichment_chunks = max_enrichment_chunks if max_enrichment_chunks is not None else _CFG_MAX_ENRICHMENT_CHUNKS
        
        self.context_token_limit = context_token_limit if context_token_limit is not None else _CFG_CTX_LIMIT
        self.top_anchor_nodes = top_anchor_nodes if top_anchor_nodes is not None else _CFG_TOP_ANCHOR_NODES
        self._llm_client = None
        if api_key:
            from src.utils.openai_client import create_openai_client
            self._llm_client = create_openai_client(api_key=api_key, base_url=base_url)
        
        self.embedding_file = str(self.persist_directory / "node_embeddings.npz")
        
        self.embedder = Embedder(
            embedder_type="openai",
            api_key=emb_api_key,
            base_url=embedding_base_url or emb_base_url,
            model_name=embedding_model,
        )
        
        self.vector_db = VectorDB(
            backend="faiss",
            index_type="Flat",
            dimension=self.embedder.dimension,
            persist_directory=str(self.persist_directory / "vector_store"),
        )
        
        self.graph_store = GraphStore(
            namespace="flow_rag",
            persist_directory=str(self.persist_directory),
        )
        
        self.graph_store.set_vector_db(self.vector_db)
        self.graph_store.set_embedder(self.embedder)
        
        _safe_print("[Pipeline] 预加载图谱到内存...")
        graph = self.graph_store.graph
        _safe_print(f"[Pipeline] 图谱预加载完成: {len(graph.nodes)} 节点, {len(graph.edges)} 边")
        
        self.keyword_index = KeywordIndex(
            embedder=self.embedder,
            persist_path=str(self.persist_directory / "keyword_index"),
        )
        
        self.keyword_extractor = LocalKeywordExtractor(
            top_k=5,
            embedder=self.embedder,
            use_mmr=True,
            mmr_diversity=0.3,
        )
        
        self.graph_retriever = GraphRetriever(
            graph_store=self.graph_store,
            embedder=self.embedder,
            expand_hops=expand_hops,
            include_causal_chain=include_causal_chain,
        )
        
        self.vector_retriever = VectorRetriever(
            vector_db=self.vector_db,
            embedder=self.embedder,
        )
    
    def _call_llm_with_retry(self, messages, temperature=0.6,
                              max_retries=None, caller="unknown") -> str:
        """带重试的 LLM 调用，自动处理空响应和 429 限流"""
        from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS
        _max = API_RETRY_MAX_ATTEMPTS if max_retries is None else max_retries
        _interval = API_RETRY_INTERVAL_SECONDS
        last_error = None
        for attempt in range(_max + 1):
            try:
                response = self._llm_client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=messages,
                    temperature=temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False}
                )
                
                content = _safe_get_content(response)
                
                if content and content.strip():
                    if attempt > 0:
                        _safe_print(f"[LLM重试] {caller}: 第{attempt+1}次尝试成功", verbose=True)
                    return content
                
                _safe_print(f"[LLM空响应] {caller}: 第{attempt+1}/{_max+1}次，content为空", verbose=True)
                if response and hasattr(response, 'choices') and response.choices:
                    msg = response.choices[0].message
                    _safe_print(f"  message.content: {repr(msg.content) if msg else 'None'}", verbose=True)
                    if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                        _safe_print(f"  message.reasoning_content: {msg.reasoning_content[:200]}...", verbose=True)
                
                last_error = EmptyResponseError(f"{caller}: LLM返回空content (第{attempt+1}次)")
                
            except Exception as e:
                _safe_print(f"[LLM异常] {caller}: 第{attempt+1}/{_max+1}次, 错误: {e}")
                last_error = e
            
            if attempt < _max:
                _safe_print(f"[LLM重试] {caller}: 等待 {_interval}s 后重试...", verbose=True)
                time.sleep(_interval)
        
        raise last_error or EmptyResponseError(f"{caller}: LLM返回空content，已重试{_max+1}次")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        node_types: Optional[List[str]] = None,
        exploration_strategy: str = "full_context",
    ) -> HybridResult:
        """执行检索
        
        Args:
            query: 用户问题
            top_k: 检索数量
            mode: "hybrid" / "graph" / "reasoning"
            exploration_strategy: 探索策略 "step_by_step" / "full_context"（仅 reasoning 模式）
        """
        node_type_enums = None
        if node_types:
            node_type_enums = [NodeType(nt) for nt in node_types]
        
        if mode == "hybrid":
            return self._retrieve_hybrid(query, top_k, node_type_enums)
        
        elif mode == "graph":
            graph_results = self.graph_retriever.retrieve(query, top_k=top_k, node_types=node_type_enums)
            return HybridResult(
                query=query,
                reasoning_result=ReasoningResult(
                    query=query,
                    graph_results=graph_results,
                    graph_recall_count=len(graph_results),
                ),
                fusion_strategy="graph_only",
            )
        
        elif mode == "reasoning":
            return self._retrieve_with_reasoning(
                query, top_k, node_type_enums,
                max_iterations=self.max_reasoning_iterations,
                exploration_strategy=exploration_strategy,
            )
        
        else:
            raise ValueError(f"不支持的检索模式: {mode}，支持: hybrid/graph/reasoning")
    
    def _retrieve_by_entity(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: np.ndarray = None,
    ) -> List[RetrievalResult]:
        if query_embedding is None:
            query_embedding = self.embedder.embed_query(query)

        results = []
        seen_chunk_ids = set()
        
        extraction_result = self.keyword_extractor.extract(query)
        query_keywords = extraction_result.keywords
        
        _safe_print(f"  问题提取的关键词: {query_keywords}", verbose=True)
        
        if not query_keywords or self.keyword_index.count() == 0:
            _safe_print(f"  无关键词或索引为空，返回空结果", verbose=True)
            return results
        
        matched_keywords_map: dict = {}  # chunk_id -> matched_keyword
        
        for query_kw in query_keywords:
            similar_keywords = self.keyword_index.search_by_text(query_kw, top_k=5)
            for keyword, score in similar_keywords:
                chunk_ids = self.keyword_index.get_chunks_by_keyword(keyword)
                for chunk_id in chunk_ids:
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        matched_keywords_map[chunk_id] = keyword
        
        _safe_print(f"  找到 {len(seen_chunk_ids)} 个关联知识块", verbose=True)
        
        for chunk_id in seen_chunk_ids:
            chunk = self.vector_db.get_chunk(chunk_id)
            if chunk:
                text_embedding = self.vector_db.get_chunk_embedding(chunk_id)
                if text_embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, text_embedding)
                else:
                    text_embedding = self.embedder.embed_query(chunk.text)
                    similarity = self._cosine_similarity(query_embedding, text_embedding)
                
                results.append(RetrievalResult(
                    id=chunk_id,
                    text=chunk.text,
                    score=similarity,
                    source_type="keyword",
                    metadata={
                        "source_file": chunk.source_file,
                        "matched_keyword": matched_keywords_map.get(chunk_id, ""),
                        "knowledge_type": chunk.metadata.get("knowledge_type", ""),
                    }
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        final_results = results[:top_k]
        
   
        return final_results

    # ============================================================
    # 问题类型检测 + 类型感知 Prompt
    # ============================================================
    @staticmethod
    def _detect_question_type(query: str) -> str:
        """检测问题类型: factual / procedural / conditional / causal"""
        q = query.strip()
        # 流程/步骤型（强信号关键词）
        if any(kw in q for kw in [
            "步骤", "流程", "怎么做", "如何操作", "操作方法", "怎样进行",
            "怎么进行", "执行顺序", "操作顺序", "程序",
            "需要经过", "需要哪些步骤", "如何执行", "怎么执行",
        ]):
            return "procedural"
        # 条件/分支型
        if any(kw in q for kw in [
            "如果", "若", "什么情况下", "什么条件",
            "不满足", "异常时", "故障时",
            "区别", "不同",
        ]):
            return "conditional"
        # 因果型（仅保留强因果信号，去掉"结果""影响"等弱信号）
        if any(kw in q for kw in [
            "为什么", "原因", "导致", "造成", "引起",
            "后果", "为何", "怎么会",
        ]):
            return "causal"
        # 默认：事实型
        return "factual"
    
    @staticmethod
    def _build_answer_instructions(query: str, question_type: str) -> str:
        """根据问题类型构建答案生成指令（第1条按题型变化，其余通用）"""
        if question_type == "procedural":
            echo_rule = "首句复述问题核心词并概括操作目标"
        elif question_type == "conditional":
            echo_rule = "首句复述问题核心词并概括回答"
        elif question_type == "causal":
            echo_rule = "首句复述问题核心词并给出核心原因/结果"
        else:  # factual
            echo_rule = "首句复述问题核心词并直接给出答案（如问'XX上限是多少'→ 答'XX上限为70℃'）"
        
        return (
            f"要求：\n"
            f"1. {echo_rule}\n"
            f"2. 简洁精准：事实型问题用1-3句话回答；流程型问题列出关键步骤\n"
            f"3. 如果涉及步骤，用①②③编号按顺序列出\n"
            f"4. 如果涉及条件，写明'当...时，执行...'\n"
            f"5. 数值必须带单位和完整语境（如'弯曲半径≥50mm'）\n"
            f"6. 基于参考资料原文作答，覆盖关键细节，用词尽量贴近原文表述\n"
            f"7. 只回答问题直接询问的内容，不要补充额外知识、延伸分析或背景介绍\n"
            f"8. 禁止：不要提及知识图谱、推理路径、节点、参考编号等系统内部信息\n"
            f"9. 禁止：不要写'根据知识X''基于路径''逻辑依据'等元描述\n"
            f"10. 仅当参考资料完全无关时，才回答'无法回答'"
        )
    
    def generate_answer(
        self,
        query: str,
        retrieval_result: HybridResult,
    ) -> str:
        """根据检索结果生成答案（reasoning 模式直接返回已有答案，hybrid 模式调用 LLM）"""
        if not self._llm_client:
            return "无法生成答案：LLM 客户端未初始化"
        
        if retrieval_result.reasoning_result:
            agent_answer = retrieval_result.reasoning_result.agent_answer or ""
            if agent_answer and agent_answer.strip() and "无法回答" not in agent_answer:
                _safe_print(f"[生成答案] 使用 Agent 推理答案（无需额外 LLM 调用）")
                return agent_answer
        
        _safe_print(f"[生成答案] 基于检索结果生成答案...")
        
        # 图谱/向量分值量纲不同，按 rank 交替合并而非按原始分值排序
        g_sorted = sorted(retrieval_result.graph_results, key=lambda x: x.score, reverse=True)
        v_sorted = sorted(retrieval_result.vector_results, key=lambda x: x.score, reverse=True)
        all_results = []
        seen_ids = set()
        gi, vi = 0, 0
        while gi < len(g_sorted) or vi < len(v_sorted):
            if gi < len(g_sorted):
                r = g_sorted[gi]; gi += 1
                if r.id not in seen_ids:
                    seen_ids.add(r.id); all_results.append(r)
            if vi < len(v_sorted):
                r = v_sorted[vi]; vi += 1
                if r.id not in seen_ids:
                    seen_ids.add(r.id); all_results.append(r)
        
        context_parts = []
        accumulated_tokens = 0
        PROMPT_OVERHEAD_TOKENS = 250
        SEPARATOR_TOKENS = 6
        limit = max(0, getattr(self, 'context_token_limit', 7000) - PROMPT_OVERHEAD_TOKENS)
        
        for result in all_results:
            text = (result.text or "").strip()
            if not text:
                continue
            
            item_tokens = self._count_tokens(text)
            sep_cost = SEPARATOR_TOKENS if context_parts else 0
            
            if accumulated_tokens + item_tokens + sep_cost <= limit:
                context_parts.append(text)
                accumulated_tokens += item_tokens + sep_cost
            else:
                _safe_print(f"[生成答案] 上下文达到 token 限制 ({accumulated_tokens}/{limit})，停止添加知识", verbose=True)
                break
        
        context = "\n---\n".join(context_parts)
        
        if not context_parts:
            return "无法回答：未检索到相关参考资料。"
        
        q_type = self._detect_question_type(query)
        instructions = self._build_answer_instructions(query, q_type)
        
        prompt = f"""你是工业设备技术专家。请仅根据以下参考资料直接回答问题，不要编造信息。

<参考资料>
{context}
</参考资料>

<问题>
{query}
</问题>

{instructions}

回答："""
        
        _safe_print(f"[LLM调用] generate_answer (题型: {q_type})", verbose=True)
        
        try:
            answer = self._call_llm_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_retries=3,
                caller="generate_answer"
            )
            
            _safe_print(f"[LLM原始输出] {repr(answer[:300]) if answer else '(空)'}", verbose=True)
            
            return answer.strip()
            
        except EmptyResponseError as e:
            _safe_print(f"[LLM错误-空响应] generate_answer: {e}")
            raise
            
        except Exception as e:
            _safe_print(f"[LLM错误] generate_answer: {str(e)}")
            raise
    
    def retrieve_and_generate(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "reasoning",
        exploration_strategy: str = "full_context",
    ) -> Dict[str, Any]:
        """一站式接口：检索 + 推理 + 生成答案"""
        _safe_print(f"\n{'='*50}", verbose=True)
        _safe_print(f"问题: {query}", verbose=True)
        _safe_print(f"{'='*50}", verbose=True)
        
        retrieval_result = self.retrieve(query, top_k=top_k, mode=mode, exploration_strategy=exploration_strategy)
        
        if (mode == "reasoning"):
            agent_answer = retrieval_result.reasoning_result.agent_answer or ""
            final_ctx = getattr(retrieval_result.reasoning_result, 'final_context', None)
            
            graph_paths = []
            if retrieval_result.reasoning_result and retrieval_result.reasoning_result.reasoning_paths:
                for path in retrieval_result.reasoning_result.reasoning_paths:
                    if len(path.path_nodes) > 1:
                        graph_paths.append(" → ".join(path.path_nodes))
            
            if final_ctx and len(final_ctx) > 0:
                _safe_print(f"[生成] 基于完整上下文（{len(final_ctx)}条知识 + {len(graph_paths)}条推理路径）生成答案...", verbose=True)
                answer = self._generate_final_answer(
                    query=query,
                    knowledge_list=final_ctx,
                    graph_reasoning_paths=graph_paths,
                )
                _safe_print(f"[最终答案] {repr(answer[:300])}", verbose=True)
            elif agent_answer and agent_answer.strip():
                answer = agent_answer
            else:
                _safe_print(f"[生成] Agent 推理答案为空，fallback 到 generate_answer...", verbose=True)
                answer = self.generate_answer(query, retrieval_result)
        else:
            answer = self.generate_answer(query, retrieval_result)
        
        reasoning_paths = []
        if retrieval_result.reasoning_result and retrieval_result.reasoning_result.reasoning_paths:
            for path in retrieval_result.reasoning_result.reasoning_paths:
                if len(path.path_nodes) > 1:
                    reasoning_paths.append(" → ".join(path.path_nodes))
                elif len(path.path_nodes) == 1:
                    reasoning_paths.append(f"[单节点] {path.path_nodes[0]}")

        
        _safe_print(f"\n{'='*50}", verbose=True)
        _safe_print(f"【答案】{answer}", verbose=True)
        _safe_print(f"{'='*50}", verbose=True)
        
        if reasoning_paths:
            _safe_print(f"【推理路径】", verbose=True)
            for i, path in enumerate(reasoning_paths, 1):
                _safe_print(f"  {i}. {path}", verbose=True)
        
        return {
            "query": query,
            "answer": answer,
            "reasoning_paths": reasoning_paths,
            "retrieval_result": retrieval_result,
        }
    
    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        node_types: Optional[List[NodeType]] = None,
    ) -> HybridResult:
        """两路融合检索：实体检索 + 图谱检索（并行执行，RRF 融合）"""
        graph_results: List[RetrievalResult] = []
        entity_results: List[RetrievalResult] = []
        
        def do_entity_retrieval():
            return self._retrieve_by_entity(query, top_k)
        
        def do_graph_retrieval():
            return self.graph_retriever.retrieve(
                query,
                top_k=top_k,
                node_types=node_types,
                expand_results=True,
            )
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(do_entity_retrieval): "entity",
                executor.submit(do_graph_retrieval): "graph",
            }
            
            for future in as_completed(futures):
                retrieval_type = futures[future]
                try:
                    results = future.result()
                    if retrieval_type == "entity":
                        entity_results = results
                    else:
                        graph_results = results
                except Exception as e:
                    _safe_print(f"[RetrievalPipeline] {retrieval_type}检索出错: {e}")
        
        merged_results = self._two_way_fusion(
            graph_results, entity_results, top_k
        )
        
        return HybridResult(
            query=query,
            reasoning_result=ReasoningResult(
                query=query,
                graph_results=graph_results,
                vector_results=entity_results,
                final_context=merged_results,
                graph_recall_count=len(graph_results),
                vector_recall_count=len(entity_results),
            ),
            fusion_strategy="entity_graph_rrf",
        )
    
    
    def _two_way_fusion(
        self,
        graph_results: List[RetrievalResult],
        entity_results: List[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """两路 RRF 融合（图谱 + 实体）"""
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, RetrievalResult] = {}
        
        for rank, result in enumerate(graph_results):
            score = self.graph_weight / (k + rank + 1)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + score
            if result.id not in result_map:
                result_map[result.id] = result
        
        for rank, result in enumerate(entity_results):
            score = self.entity_weight / (k + rank + 1)
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + score
            if result.id not in result_map:
                result_map[result.id] = result
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        merged = []
        for result_id in sorted_ids[:top_k]:
            result = result_map[result_id]
            merged.append(RetrievalResult(
                id=result.id,
                text=result.text,
                score=rrf_scores[result_id],
                source_type=result.source_type,
                metadata={
                    **result.metadata,
                    "fusion_method": "entity_graph_rrf",
                }
            ))
        
        return merged
    
    # ========== 迭代式推理检索 ==========
    
    def _retrieve_with_reasoning(
        self,
        query: str,
        top_k: int,
        node_types: Optional[List[NodeType]] = None,
        max_iterations: int = 5,
        exploration_strategy: str = "full_context",
    ) -> HybridResult:
        """迭代式推理检索：锚点识别 -> 多跳探索 -> 知识收集 -> 答案生成"""
        
        core_event_info = self._extract_core_event(query)
        system_device = core_event_info.get("system_device", "")
        core_event = core_event_info.get("core_event", query)
        
        _safe_print(f"[Agent] 核心事件: {core_event}", verbose=True)
        
        query_embedding = self.embedder.embed_query(query)
        
        collected_knowledge: Dict[str, RetrievalResult] = {}
        collected_chunk_ids: set = set()
        explored_nodes: set = set()
        current_frontier: List[str] = []
        all_reasoning_paths: List[ReasoningPath] = []
        iteration_count = 0
        reasoning_trace: List[Dict] = []
        accumulated_tokens = 0
        token_limit_reached = False
        
        _safe_print(f"[Agent] Step 1: 使用核心事件寻找锚点节点...", verbose=True)
        anchor_results = self.graph_store.search_nodes_vector_then_keywords(
            system_device=system_device,
            query_embedding=query_embedding,
            origin_query=query,
            top_k=top_k,
            node_types=node_types,
            recall_multiplier=3,
        )
    
  
        _safe_print(f"[Agent] 找到 {len(anchor_results)} 个锚点节点", verbose=True)
        
        winning_anchor = None
        judgment = {"sufficient": False}
        
        for anchor_idx, anchor in enumerate(anchor_results[:self.top_anchor_nodes]):
            # 检查是否已达到 token 限制
            if token_limit_reached:
                _safe_print(f"[Agent] Token 限制已达到，停止探索后续锚点", verbose=True)
                break
            
            # 动态预算分配：靠前锚点获得更多预算（1.4x 倾斜）
            remaining_anchors = self.top_anchor_nodes - anchor_idx
            remaining_budget = self.context_token_limit - accumulated_tokens
            per_anchor_budget = max(800, int(remaining_budget / remaining_anchors * 1.4))
            anchor_token_limit = min(self.context_token_limit, accumulated_tokens + per_anchor_budget)
            
            _safe_print(f"[Agent] 探索锚点 {anchor_idx + 1}/{self.top_anchor_nodes}: {anchor.id} (预算:{per_anchor_budget}t)", verbose=True)
            
            if exploration_strategy == "full_context":
                explore_result = self._explore_anchor_with_full_context(
                    anchor=anchor,
                    anchor_idx=anchor_idx,
                    query=query,
                    max_iterations=max_iterations,
                    accumulated_tokens=accumulated_tokens,
                    reasoning_trace=reasoning_trace,
                    all_reasoning_paths=all_reasoning_paths,
                    query_embedding=query_embedding,
                    anchor_token_limit=anchor_token_limit,
                    collected_chunk_ids=collected_chunk_ids,
                )
            else:
                explore_result = self._explore_anchor_step_by_step(
                    anchor=anchor,
                    anchor_idx=anchor_idx,
                    query=query,
                    max_iterations=max_iterations,
                    accumulated_tokens=accumulated_tokens,
                    reasoning_trace=reasoning_trace,
                    all_reasoning_paths=all_reasoning_paths,
                    query_embedding=query_embedding,
                    anchor_token_limit=anchor_token_limit,
                    collected_chunk_ids=collected_chunk_ids,
                )
            
            anchor_knowledge = explore_result["anchor_knowledge"]
            anchor_chunk_ids = explore_result["anchor_chunk_ids"]
            anchor_explored = explore_result["anchor_explored"]
            judgment = explore_result["judgment"]
            accumulated_tokens = explore_result["accumulated_tokens"]
            token_limit_reached = explore_result["token_limit_reached"]
            iteration_count += explore_result.get("iteration_count", 0)
            
            collected_chunk_ids.update(anchor_chunk_ids)
            
            if token_limit_reached:
                winning_anchor = anchor
                for k, v in anchor_knowledge.items():
                    if k not in collected_knowledge:
                        collected_knowledge[k] = v
                explored_nodes.update(anchor_explored)
                
                if accumulated_tokens >= self.context_token_limit:
                    _safe_print(f"[Agent] 全局 token 预算耗尽，停止探索", verbose=True)
                    break
                else:
                    token_limit_reached = False
                    _safe_print(f"[Agent] 锚点预算用完，切换到下一锚点", verbose=True)
                    continue
            
            if judgment.get("sufficient", False):
                winning_anchor = anchor
                for k, v in anchor_knowledge.items():
                    if k not in collected_knowledge:
                        collected_knowledge[k] = v
                explored_nodes.update(anchor_explored)
                
                remaining_ratio = 1 - (accumulated_tokens / self.context_token_limit) if self.context_token_limit > 0 else 0
                if remaining_ratio < 0.15:
                    _safe_print(f"[Agent] 已找到答案，预算几乎耗尽，停止探索", verbose=True)
                    break
                else:
                    _safe_print(f"[Agent] 已找到答案，继续探索下一锚点 (剩余 {remaining_ratio:.0%})", verbose=True)
                    continue
            
            if judgment.get("skip_anchor", False):
                _safe_print(f"[Agent] 跳过此锚点，继续下一个锚点", verbose=True)
            
            for k, v in anchor_knowledge.items():
                if k not in collected_knowledge:
                    collected_knowledge[k] = v
            explored_nodes.update(anchor_explored)

        if not judgment["sufficient"]:
            _safe_print(f"[Agent] 所有锚点探索完毕，合并知识（共 {len(collected_knowledge)} 条）", verbose=True)
        
        _safe_print(f"[Agent] 推理完成，共 {len(all_reasoning_paths)} 条路径", verbose=True)
        
        is_sufficient = judgment.get("sufficient", False) if isinstance(judgment, dict) else False
        agent_answer = ""
        agent_reasoning = ""
        supplemental_results: List[RetrievalResult] = []
        
        if isinstance(judgment, dict):
            agent_answer = str(judgment.get("answer", "")) if judgment.get("answer") else ""
            agent_reasoning = str(judgment.get("reason", "")) if judgment.get("reason") else ""
        
        need_supplement = (
            not is_sufficient or 
            not agent_answer or 
            agent_answer.strip() == "" or
            "无法回答" in agent_answer
        )
        
        if need_supplement:
            _safe_print(f"[Agent] Step 4: 补充知识...", verbose=True)
            gap_results = self._retrieve_by_entity(query, top_k=20, query_embedding=query_embedding)
            
            if gap_results:
                remaining_tokens = max(0, self.context_token_limit - accumulated_tokens)
                used_gap_results = []
                current_context_tokens = 0
                for r in gap_results:
                    if not r.text:
                        continue
                    item_tokens = self._count_tokens(r.text)
                    
                    if current_context_tokens + item_tokens <= remaining_tokens:
                        used_gap_results.append(r)
                        current_context_tokens += item_tokens
                    else:
                        break
                
                for r in used_gap_results:
                    if r.id not in collected_knowledge:
                        collected_knowledge[r.id] = r
                
                accumulated_tokens += current_context_tokens

        final_context = list(collected_knowledge.values())
        
        # 向量补充上限为总预算的 35%，避免挤占图谱探索结果
        existing_ids = set(collected_knowledge.keys())
        VECTOR_SUPPLEMENT_RATIO = 0.35
        vector_supplement_cap = int(self.context_token_limit * VECTOR_SUPPLEMENT_RATIO)
        remaining_budget = min(self.context_token_limit - accumulated_tokens, vector_supplement_cap)
        
        if is_sufficient and accumulated_tokens > 0:
            supplement_cap = int(self.context_token_limit * 0.20)
        else:
            supplement_cap = remaining_budget
        effective_supplement_budget = min(remaining_budget, supplement_cap)
        
        if effective_supplement_budget > 200:
            _safe_print(f"[Agent] Step 5.5: 向量补充检索 (预算 {effective_supplement_budget} tokens)", verbose=True)
            vector_supplement = self.vector_retriever.retrieve(query, top_k=150)
            supplement_added = 0
            supplement_tokens_used = 0
            skipped_existing = 0
            skipped_too_short = 0
            for vs_result in vector_supplement:
                if vs_result.id in existing_ids:
                    skipped_existing += 1
                    continue
                
                if vs_result.text:
                    vs_result.text = _re.sub(r' {2,}', ' ', vs_result.text)
                
                current_remaining = effective_supplement_budget - supplement_tokens_used
                if current_remaining <= 0:
                    break
                
                chunk_tokens = self._count_tokens(vs_result.text)
                
                if chunk_tokens < 100:
                    skipped_too_short += 1
                    continue
                
                if chunk_tokens > current_remaining:
                    if current_remaining < 50:
                        break
                    truncated_text = self._truncate_to_tokens(vs_result.text, current_remaining)
                    vs_result.text = truncated_text
                    chunk_tokens = self._count_tokens(truncated_text)
                
                vs_result.source_type = "vector_supplement"
                final_context.append(vs_result)
                existing_ids.add(vs_result.id)
                accumulated_tokens += chunk_tokens
                supplement_tokens_used += chunk_tokens
                supplement_added += 1
            
            _safe_print(f"[Agent] Step 5.5 完成: 新增 {supplement_added} 条 ({supplement_tokens_used} tokens)", verbose=True)
            if supplement_added > 0:
                supplemental_results.extend([r for r in final_context if r.source_type == "vector_supplement"])

        # graph_node 条目仅含节点名称，移除后重新按 query 相似度排序
        final_context = [r for r in final_context if r.source_type != "graph_node"]
        for result in final_context:
            cached_embedding = self.vector_db.get_chunk_embedding(result.id)
            if cached_embedding is not None:
                result.score = self._cosine_similarity(query_embedding, cached_embedding)
            elif result.text:
                text_embedding = self.embedder.embed_query(result.text)
                result.score = self._cosine_similarity(query_embedding, text_embedding)
            else:
                result.score = 0.0
        final_context.sort(key=lambda x: x.score, reverse=True)
        
        reasoning_result = ReasoningResult(
            query=query,
            graph_results=anchor_results,
            vector_results=supplemental_results,
            final_context=final_context,
            agent_answer=agent_answer,
            agent_reasoning=agent_reasoning,
            anchor_nodes=anchor_results,
            reasoning_paths=all_reasoning_paths,
            total_iterations=iteration_count,
            explored_nodes_count=len(explored_nodes),
            graph_recall_count=len(anchor_results),
            vector_recall_count=len(supplemental_results),
            total_tokens=accumulated_tokens,
        )
        
        _safe_print(f"[Agent] 完成! 迭代{iteration_count}次, 探索{len(explored_nodes)}节点, 收集{len(final_context)}条知识", verbose=True)
        
        return HybridResult(
            query=query,
            reasoning_result=reasoning_result,
            fusion_strategy="agent_reasoning",
        )
    
    def _generate_final_answer(
        self,
        query: str,
        knowledge_list: List[RetrievalResult],
        graph_reasoning_paths: Optional[List[str]] = None,
    ) -> str:
        """基于收集的知识和图推理路径生成最终答案（推理路径引导 LLM 按因果/流程顺序组织回答）"""
        if not self._llm_client:
            return "无法生成答案：LLM 客户端未初始化"
        
        # 分源组装：图谱知识用宽松阈值优先填充，向量补充用严格阈值填充剩余空间
        GRAPH_FLOOR = 0.25
        VECTOR_FLOOR = 0.40
        GRAPH_BUDGET_RATIO = 0.70
        
        PROMPT_OVERHEAD_TOKENS = 350
        token_budget = max(0, getattr(self, 'context_token_limit', 7000) - PROMPT_OVERHEAD_TOKENS)
        graph_budget = int(token_budget * GRAPH_BUDGET_RATIO)
        
        graph_items = [r for r in knowledge_list if r.source_type in ("enrichment", "graph_node")]
        vector_items = [r for r in knowledge_list if r.source_type == "vector_supplement"]
        graph_items.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        vector_items.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        context_parts = []
        context_tokens = 0
        skipped_low_relevance = 0
        SEPARATOR_TOKENS = 6
        
        for result in graph_items:
            text = (result.text or "").strip()
            if not text:
                continue
            if hasattr(result, 'score') and result.score is not None and result.score < GRAPH_FLOOR:
                skipped_low_relevance += 1
                continue
            item_tokens = self._count_tokens(text)
            sep_cost = SEPARATOR_TOKENS if context_parts else 0
            if context_tokens + item_tokens + sep_cost > graph_budget:
                remaining = graph_budget - context_tokens - sep_cost
                if remaining > 100:
                    text = self._truncate_to_tokens(text, remaining)
                    context_parts.append(text)
                    context_tokens += remaining + sep_cost
                break
            context_parts.append(text)
            context_tokens += item_tokens + sep_cost
        
        graph_used = context_tokens
        
        for result in vector_items:
            text = (result.text or "").strip()
            if not text:
                continue
            if hasattr(result, 'score') and result.score is not None and result.score < VECTOR_FLOOR:
                skipped_low_relevance += 1
                continue
            item_tokens = self._count_tokens(text)
            sep_cost = SEPARATOR_TOKENS if context_parts else 0
            if context_tokens + item_tokens + sep_cost > token_budget:
                remaining = token_budget - context_tokens - sep_cost
                if remaining > 100:
                    text = self._truncate_to_tokens(text, remaining)
                    context_parts.append(text)
                    context_tokens += remaining + sep_cost
                break
            context_parts.append(text)
            context_tokens += item_tokens + sep_cost
        
        context = "\n---\n".join(context_parts)
        if skipped_low_relevance > 0:
            _safe_print(f"[生成答案] 过滤低相关性知识: {skipped_low_relevance} 条", verbose=True)
        _safe_print(f"[生成答案] 最终上下文: {len(context_parts)} 段, {context_tokens} tokens", verbose=True)
        
        order_hint = ""
        if graph_reasoning_paths:
            all_steps = []
            for path in graph_reasoning_paths:
                for step in path.split(" → "):
                    s = step.strip()
                    if s and s not in all_steps:
                        all_steps.append(s)
            if all_steps:
                order_hint = "\n【建议回答顺序】" + " → ".join(all_steps) + "\n"
        
        q_type = self._detect_question_type(query)
        instructions = self._build_answer_instructions(query, q_type)
        
        prompt = f"""你是工业设备技术专家。请仅根据以下参考资料直接回答问题，不要编造信息。

<参考资料>
{context}
</参考资料>
{order_hint}
<问题>
{query}
</问题>

{instructions}

回答："""

        
        gen_temperature = 0.3
        
        try:
            answer = self._call_llm_with_retry(
                messages=[{"role": "user", "content": prompt}],
                temperature=gen_temperature,
                max_retries=3,
                caller="_generate_final_answer"
            )
            
            return answer.strip()
            
        except EmptyResponseError as e:
            _safe_print(f"[LLM错误-空响应] _generate_final_answer: {e}")
            raise
            
        except Exception as e:
            _safe_print(f"[LLM错误] _generate_final_answer: {str(e)}")
            raise
    
    _tiktoken_encoding = None

    @classmethod
    def _get_encoding(cls):
        if cls._tiktoken_encoding is None:
            import tiktoken
            cls._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        return cls._tiktoken_encoding

    TOKENIZER_SAFETY_FACTOR = 1.15

    def _count_tokens(self, text: str) -> int:
        """估算 token 数量（cl100k_base 对中文偏低，乘安全系数补偿）"""
        if not text:
            return 0
        raw = len(self._get_encoding().encode(text))
        return int(raw * self.TOKENIZER_SAFETY_FACTOR)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """将文本截断到指定的 token 数量"""
        if not text or max_tokens <= 0:
            return ""
        
        encoding = self._get_encoding()
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens) + "..."
    
    def _collect_node_and_enrichment(
        self,
        node_id: str,
        collected_knowledge: Optional[Dict[str, RetrievalResult]],
        collected_chunk_ids: set,
        query_embedding=None,
        max_chunks: Optional[int] = None,
        node: Optional[Any] = None,
    ) -> tuple:
        """收集节点及其挂载知识块（去重、按相似度排序），返回 (new_chunk_ids, display_str)"""
        if node is None:
            node = self.graph_store.graph.get_node(node_id)
        if not node:
            return [], ""
            
        top_k = max_chunks if max_chunks is not None else self.max_enrichment_chunks
        new_chunk_ids = []
        
        if collected_knowledge is not None and node_id not in collected_knowledge:
            collected_knowledge[node_id] = RetrievalResult(
                id=node_id, text=node.name,
                source_type="graph_node",
                metadata={"node_type": node.node_type.value},
            )
            
        if not node.enrichment_chunks:
            return new_chunk_ids, ""
            
        candidates = []
        for chunk_id in node.enrichment_chunks:
            if chunk_id not in collected_chunk_ids:
                chunk = self.vector_db.get_chunk(chunk_id) if hasattr(self, 'vector_db') else None
                if chunk:
                    candidates.append((chunk_id, chunk))
                    
        if query_embedding is not None and candidates:
            scored = []
            for chunk_id, chunk in candidates:
                chunk_embedding = self.vector_db.get_chunk_embedding(chunk_id)
                sim = self._cosine_similarity(query_embedding, chunk_embedding) if chunk_embedding is not None else 0.0
                scored.append((chunk_id, chunk, sim))
            scored.sort(key=lambda x: x[2], reverse=True)
            candidates = [(c[0], c[1]) for c in scored]
            
        selected = candidates[:top_k]
        
        for chunk_id, chunk in selected:
            new_chunk_ids.append(chunk_id)
            
            if collected_knowledge is not None and chunk_id not in collected_knowledge:
                collected_knowledge[chunk_id] = RetrievalResult(
                    id=chunk_id, text=chunk.text,
                    source_type="enrichment",
                    metadata={"source_file": chunk.source_file},
                )
                
        display = "\n".join(f"  - {ch.text or ''}" for _, ch in selected)
        return new_chunk_ids, display
    
    def _get_adjacent_nodes(self, node_id: str, direction: str = "forward") -> List[Dict]:
        """获取节点的直接邻居及边信息，direction: "forward"(后继) / "backward"(前驱)"""
        graph = self.graph_store.graph
        node = graph.get_node(node_id)
        if not node:
            return []
        
        if direction == "forward":
            neighbor_ids = node.successors
        else:
            neighbor_ids = node.predecessors
        
        results = []
        for neighbor_id in neighbor_ids:
            neighbor_node = graph.get_node(neighbor_id)
            if not neighbor_node:
                continue
            
            if direction == "forward":
                edge = graph.get_edge(node_id, neighbor_id)
            else:
                edge = graph.get_edge(neighbor_id, node_id)
            
            condition_label = ""
            edge_type = "next"
            relation_name = ""
            if edge:
                edge_type = edge.edge_type.value
                relation_name = edge.relation_name or ""
                if edge.properties and "condition" in edge.properties:
                    condition_label = edge.properties["condition"]
            
            results.append({
                "id": neighbor_id,
                "name": neighbor_node.name,
                "edge_type": edge_type,
                "relation_name": relation_name,
                "condition": condition_label,
                "direction": direction,
            })
        return results

    def _get_successors(self, node_id: str) -> List[Dict]:
        return self._get_adjacent_nodes(node_id, "forward")
    
    def _get_predecessors(self, node_id: str) -> List[Dict]:
        return self._get_adjacent_nodes(node_id, "backward")

    def _bfs_traverse(self, node_id: str, direction: str = "forward",
                      max_depth: int = 5) -> List[Dict]:
        """BFS 多跳遍历邻居，限制最大深度"""
        visited = {node_id}
        collected = []
        queue = [(node_id, 0)]
        
        while queue:
            current_id, current_hop = queue.pop(0)
            if current_hop >= max_depth:
                continue
            
            neighbors = self._get_adjacent_nodes(current_id, direction)
            for nb in neighbors:
                if nb['id'] not in visited:
                    visited.add(nb['id'])
                    nb['hop'] = current_hop + 1
                    collected.append(nb)
                    queue.append((nb['id'], current_hop + 1))
        
        return collected

    def _get_all_predecessors(self, node_id: str, max_depth: int = 5, **_kw) -> List[Dict]:
        return self._bfs_traverse(node_id, "backward", max_depth)
    
    def _get_all_successors(self, node_id: str, max_depth: int = 5, **_kw) -> List[Dict]:
        return self._bfs_traverse(node_id, "forward", max_depth)
    
    def _format_node_context(
        self, 
        node_id: str, 
        query: Optional[str] = None,
        used_chunk_ids: Optional[set] = None,
        anchor_knowledge: Optional[Dict[str, 'RetrievalResult']] = None,
        query_embedding=None,
    ) -> Dict[str, Any]:
        """格式化节点上下文并收集挂载知识"""
        node = self.graph_store.graph.get_node(node_id)
        if not node:
            return {"context": "", "new_chunk_ids": [], "knowledge_tokens": 0}
        
        if query and query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        
        new_chunk_ids, enrichment_display = self._collect_node_and_enrichment(
            node_id=node_id,
            collected_knowledge=anchor_knowledge,
            collected_chunk_ids=used_chunk_ids or set(),
            query_embedding=query_embedding,
            node=node,
        )
        
        node_name_tokens = self._count_tokens(node.name) if node else 0
        TEMPLATE_OVERHEAD = 15
        knowledge_tokens = self._count_tokens(enrichment_display) + node_name_tokens + TEMPLATE_OVERHEAD if enrichment_display else node_name_tokens + TEMPLATE_OVERHEAD
        
        context = f"""【当前节点】
{node.name}

【挂载知识】
{enrichment_display if enrichment_display else "  (无挂载知识)"}"""
        
        return {"context": context, "new_chunk_ids": new_chunk_ids, "knowledge_tokens": knowledge_tokens}
    
    def _make_hop_result(self, *, sufficient=False, next_nodes=None, answer="",
                         reason="", this_call_tokens=0, accumulated_tokens=0,
                         token_limit_reached=False, new_chunk_ids=None,
                         skip_anchor=False) -> Dict[str, Any]:
        """构造跳转决策的标准返回值"""
        return {
            "sufficient": sufficient,
            "next_nodes": next_nodes or [],
            "answer": answer,
            "reason": reason,
            "this_call_tokens": this_call_tokens,
            "accumulated_tokens": accumulated_tokens,
            "token_limit_reached": token_limit_reached,
            "new_chunk_ids": new_chunk_ids or [],
            "skip_anchor": skip_anchor,
        }

    def _call_llm_parse_json(self, prompt: str, caller: str,
                             temperature: float = 0.3) -> Optional[Dict]:
        """调用 LLM 并解析 JSON 响应，失败返回 None"""
        if not self._llm_client:
            return None
        from src.utils.openai_client import call_with_retry
        try:
            response = call_with_retry(
                lambda: self._llm_client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False}
                ),
                caller=caller
            )
        except Exception:
            return None
        
        content = _safe_get_content(response)
        _safe_print(f"[LLM原始输出-{caller}] {repr(content[:200]) if content else '(空)'}", verbose=True)
        
        if not content:
            return None
        
        clean = _re.sub(r'```(?:json)?\s*', '', content)
        clean = _re.sub(r'```\s*$', '', clean)
        start = clean.find('{')
        if start == -1:
            return None
        depth = 0
        end = -1
        for i in range(start, len(clean)):
            if clean[i] == '{':
                depth += 1
            elif clean[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        try:
            return json.loads(clean[start:end])
        except (json.JSONDecodeError, ValueError):
            _safe_print(f"[{caller}] JSON解析失败: {clean[start:end][:100]}", verbose=True)
            return None

    def _decide_next_hop(
        self,
        current_node_id: str,
        query: str,
        visited_nodes: set,
        accumulated_tokens: int = 0,
        token_limit: Optional[int] = None,
        used_chunk_ids: Optional[set] = None,
        anchor_knowledge: Optional[Dict[str, 'RetrievalResult']] = None,
        query_embedding=None,
    ) -> Dict[str, Any]:
        """逐步探索策略：展示直接邻居，由 LLM 决定下一跳"""
        ctx = self._format_node_context(
            current_node_id, query=query, used_chunk_ids=used_chunk_ids,
            anchor_knowledge=anchor_knowledge, query_embedding=query_embedding,
        )
        new_chunk_ids = ctx["new_chunk_ids"]
        this_call_tokens = ctx["knowledge_tokens"]
        new_accumulated = accumulated_tokens + this_call_tokens
        
        if token_limit and new_accumulated > token_limit:
            _safe_print(f"[Agent] Token 超限 ({new_accumulated}/{token_limit})，停止探索", verbose=True)
            return self._make_hop_result(
                this_call_tokens=this_call_tokens, accumulated_tokens=new_accumulated,
                token_limit_reached=True, new_chunk_ids=new_chunk_ids,
                reason="检索文本 Token 配额已耗尽",
            )
        
        avail_succ = [s for s in self._get_successors(current_node_id) if s['id'] not in visited_nodes]
        avail_pred = [p for p in self._get_predecessors(current_node_id) if p['id'] not in visited_nodes]
        
        all_candidates = []
        candidates_text = ""
        if avail_succ or avail_pred:
            candidates_text = "【可跳转的候选节点】\n"
            if avail_succ:
                candidates_text += "\n后继节点:\n"
                for s in avail_succ:
                    rel = s.get('relation_name', s['edge_type'])
                    if s.get('condition'):
                        rel += f', 条件满足="{s["condition"]}"'
                    candidates_text += f"  - [{s['name']}] (关系: {rel})\n"
                    all_candidates.append(s['id'])
            if avail_pred:
                candidates_text += "\n前驱节点:\n"
                for p in avail_pred:
                    rel = p.get('relation_name', p['edge_type'])
                    if p.get('condition'):
                        rel += f', 条件满足="{p["condition"]}"'
                    candidates_text += f"  - [{p['name']}] (关系: {rel})\n"
                    all_candidates.append(p['id'])
        
        if all_candidates:
            prompt = (
                f"基于图结构推理，决定下一步探索方向。\n\n"
                f"问题: {query}\n\n{ctx['context']}\n\n{candidates_text}\n\n"
                f"【判断逻辑】\n"
                f"1. 根据当前节点知识判断能否回答问题 → sufficient=true 并提供答案\n"
                f"2. 不能回答 → 在 next_nodes 中填写要跳转的候选节点ID\n\n"
                f'请用 JSON 回复: {{"sufficient": true/false, "next_nodes": ["节点ID"], '
                f'"answer": "答案", "reason": "理由"}}'
            )
        else:
            prompt = (
                f"基于图结构推理，判断是否能回答问题。\n\n"
                f"问题: {query}\n\n{ctx['context']}\n\n"
                f'请用 JSON 回复: {{"sufficient": true/false, "answer": "答案", "reason": "理由"}}'
            )
        
        _safe_print(f"[LLM调用] _decide_next_hop (节点: {current_node_id[:30]}...) "
                     f"[知识token: {this_call_tokens}, 累计: {new_accumulated}/{token_limit or '∞'}]", verbose=True)
        
        fallback_next = [s['id'] for s in avail_succ[:1]] if avail_succ else []
        
        try:
            parsed = self._call_llm_parse_json(prompt, "decide_next_hop", temperature=0.3)
            if parsed is None:
                return self._make_hop_result(
                    next_nodes=fallback_next, reason="LLM返回为空或无法解析",
                    this_call_tokens=this_call_tokens, accumulated_tokens=new_accumulated,
                    new_chunk_ids=new_chunk_ids,
                )
            raw_next = parsed.get("next_nodes", [])
            valid_next = [n for n in raw_next if n in all_candidates]
            if not valid_next and raw_next:
                candidate_set = set(all_candidates)
                for n in raw_next:
                    stripped = n.strip("[]【】").strip()
                    if stripped in candidate_set:
                        valid_next.append(stripped)
            parsed["next_nodes"] = valid_next
            parsed.setdefault("sufficient", False)
            parsed["this_call_tokens"] = this_call_tokens
            parsed["accumulated_tokens"] = new_accumulated
            parsed["token_limit_reached"] = False
            parsed["new_chunk_ids"] = new_chunk_ids
            return parsed
        except Exception as e:
            _safe_print(f"[Agent] LLM调用出错: {e}")
            return self._make_hop_result(
                next_nodes=fallback_next, reason=f"LLM调用出错: {e}",
                this_call_tokens=this_call_tokens, accumulated_tokens=new_accumulated,
                new_chunk_ids=new_chunk_ids,
            )
    
    def _format_full_graph_context(
        self, 
        node_id: str, 
        query: Optional[str] = None,
        used_chunk_ids: Optional[set] = None,
        anchor_knowledge: Optional[Dict[str, 'RetrievalResult']] = None,
        query_embedding=None,
    ) -> Dict[str, Any]:
        """格式化节点上下文，并通过 BFS 返回所有前驱/后继供 LLM 决策"""
        node = self.graph_store.graph.get_node(node_id)
        if not node:
            return {"context": "", "new_chunk_ids": [], "all_predecessors": [], "all_successors": []}
        
        if query and query_embedding is None:
            query_embedding = self.embedder.embed_query(query)
        
        new_chunk_ids, enrichment_display = self._collect_node_and_enrichment(
            node_id=node_id,
            collected_knowledge=anchor_knowledge,
            collected_chunk_ids=used_chunk_ids or set(),
            query_embedding=query_embedding,
            node=node,
        )

        node_name_tokens = self._count_tokens(node.name) if node else 0
        FULL_CTX_TEMPLATE = 20
        knowledge_tokens = self._count_tokens(enrichment_display) + node_name_tokens + FULL_CTX_TEMPLATE if enrichment_display else node_name_tokens + FULL_CTX_TEMPLATE
        
        all_predecessors = self._get_all_predecessors(node_id)
        all_successors = self._get_all_successors(node_id)
        
        all_predecessors.sort(key=lambda x: x.get('hop', 1))
        all_successors.sort(key=lambda x: x.get('hop', 1))
        
        context = f"""【当前节点】
{node.name}

【挂载知识】
{enrichment_display if enrichment_display else "  (无挂载知识)"}
"""
        
        return {
            "context": context, 
            "new_chunk_ids": new_chunk_ids,
            "knowledge_tokens": knowledge_tokens,
            "all_predecessors": all_predecessors,
            "all_successors": all_successors
        }
    
    def _decide_next_hop_with_full_context(
        self,
        current_node_id: str,
        query: str,
        visited_nodes: set,
        accumulated_tokens: int = 0,
        token_limit: Optional[int] = None,
        used_chunk_ids: Optional[set] = None,
        anchor_knowledge: Optional[Dict[str, 'RetrievalResult']] = None,
        query_embedding=None,
    ) -> Dict[str, Any]:
        """全上下文探索策略：展示完整图结构，在直接邻居中选择下一跳"""
        ctx = self._format_full_graph_context(
            current_node_id, query=query, used_chunk_ids=used_chunk_ids,
            anchor_knowledge=anchor_knowledge,
            query_embedding=query_embedding,
        )
        new_chunk_ids = ctx["new_chunk_ids"]
        this_call_tokens = ctx["knowledge_tokens"]
        new_accumulated = accumulated_tokens + this_call_tokens
        
        if token_limit and new_accumulated > token_limit:
            _safe_print(f"[Agent] Token 超限 ({new_accumulated}/{token_limit})，停止探索", verbose=True)
            return self._make_hop_result(
                skip_anchor=True, this_call_tokens=this_call_tokens,
                accumulated_tokens=new_accumulated, token_limit_reached=True,
                new_chunk_ids=new_chunk_ids, reason="检索文本 Token 配额已耗尽",
            )
        
        # 只从直接邻居中选下一跳，避免单次迭代跨多跳导致 max_iterations 语义失真
        avail_succ = [s for s in self._get_successors(current_node_id) if s['id'] not in visited_nodes]
        avail_pred = [p for p in self._get_predecessors(current_node_id) if p['id'] not in visited_nodes]
        
        all_candidates = []
        candidates_text = ""
        if avail_succ or avail_pred:
            if avail_succ:
                candidates_text += "\n直接后继:\n"
                for s in avail_succ[:5]:
                    rel = s.get('relation_name', s['edge_type'])
                    cond = f', 条件="{s["condition"]}"' if s.get('condition') else ""
                    candidates_text += f"  - [{s['name']}] (关系: {rel}{cond})\n"
                    all_candidates.append(s['id'])
            if avail_pred:
                candidates_text += "\n直接前驱:\n"
                for p in avail_pred[:5]:
                    rel = p.get('relation_name', p['edge_type'])
                    cond = f', 条件="{p["condition"]}"' if p.get('condition') else ""
                    candidates_text += f"  - [{p['name']}] (关系: {rel}{cond})\n"
                    all_candidates.append(p['id'])
        
        answer_constraint = (
            "**答案约束：首句复述问题关键词并给出答案，数值带单位，"
            "步骤用①②③编号，条件用\"当...时→...\"格式，"
            "禁止提及节点、图谱、路径等系统术语**"
        )
        
        if all_candidates:
            prompt = (
                f"基于图结构推理，决定下一步探索方向。\n\n问题: {query}\n\n"
                f"{ctx['context']}\n\n{candidates_text}\n\n"
                f"【判断逻辑】\n"
                f"1. 判断已收集的知识能否**完整**回答问题\n"
                f"   - sufficient=true: 已有全部关键信息\n   - {answer_constraint}\n"
                f"   - 只有部分信息时应继续探索\n"
                f"2. sufficient=false 时选择最相关候选填入 next_nodes\n\n"
                f'请用 JSON 回复: {{"sufficient": true/false, "next_nodes": ["节点ID"], '
                f'"answer": "答案", "reason": "理由"}}\n'
                f"注意：next_nodes 中的ID必须来自上面列出的候选"
            )
        else:
            prompt = (
                f"基于图结构推理，判断是否能回答问题。\n\n问题: {query}\n\n"
                f"{ctx['context']}\n\n"
                f"【判断逻辑】\n"
                f"1. 能回答 → sufficient=true 并提供答案\n   {answer_constraint}\n"
                f"2. 不能回答 → sufficient=false，系统将跳转到下一个锚点\n\n"
                f'请用 JSON 回复: {{"sufficient": true/false, "answer": "答案", "reason": "理由"}}'
            )
        
        _safe_print(f"[LLM调用] full_context_hop (节点: {current_node_id[:30]}...) "
                     f"[累计: {new_accumulated}/{token_limit or '∞'}]", verbose=True)
        
        try:
            parsed = self._call_llm_parse_json(prompt, "full_context_hop", temperature=0.3)
            if parsed is None:
                return self._make_hop_result(
                    skip_anchor=True, reason="LLM返回为空或无法解析",
                    this_call_tokens=this_call_tokens, accumulated_tokens=new_accumulated,
                    new_chunk_ids=new_chunk_ids,
                )
            raw_next = parsed.get("next_nodes", [])
            valid_next = [n for n in raw_next if n in all_candidates]
            if not valid_next and raw_next:
                candidate_set = set(all_candidates)
                for n in raw_next:
                    stripped = n.strip("[]【】").strip()
                    if stripped in candidate_set:
                        valid_next.append(stripped)
            is_sufficient = parsed.get("sufficient", False)
            parsed["next_nodes"] = valid_next
            parsed["skip_anchor"] = (not is_sufficient) and (not valid_next)
            parsed["this_call_tokens"] = this_call_tokens
            parsed["accumulated_tokens"] = new_accumulated
            parsed["token_limit_reached"] = False
            parsed["new_chunk_ids"] = new_chunk_ids
            _safe_print(f"[LLM输出] sufficient={is_sufficient}, next={valid_next}, skip={parsed['skip_anchor']}", verbose=True)
            return parsed
        except Exception as e:
            _safe_print(f"[Agent] LLM调用出错: {e}")
            return self._make_hop_result(
                skip_anchor=True, reason=f"LLM调用出错: {e}",
                this_call_tokens=this_call_tokens, accumulated_tokens=new_accumulated,
                new_chunk_ids=new_chunk_ids,
            )
    
    def _explore_anchor(
        self,
        anchor: 'RetrievalResult',
        anchor_idx: int,
        query: str,
        max_iterations: int,
        accumulated_tokens: int,
        reasoning_trace: List[Dict],
        all_reasoning_paths: List['ReasoningPath'],
        query_embedding=None,
        anchor_token_limit: int = None,
        collected_chunk_ids: set = None,
        strategy: str = "full_context",
    ) -> Dict[str, Any]:
        """从锚点出发沿图多跳探索，strategy: "step_by_step" / "full_context" """
        use_full_ctx = (strategy == "full_context")
        decide_fn = self._decide_next_hop_with_full_context if use_full_ctx else self._decide_next_hop

        anchor_knowledge: Dict[str, RetrievalResult] = {}
        anchor_chunk_ids: set = set(collected_chunk_ids) if collected_chunk_ids else set()
        anchor_explored: set = set()
        current_node_id = anchor.id
        anchor_explored.add(current_node_id)

        effective_token_limit = anchor_token_limit if anchor_token_limit is not None else self.context_token_limit

        token_limit_reached = False
        judgment: Dict[str, Any] = {"sufficient": False}
        iteration_count = 0
        final_answer = ""

        anchor_node = self.graph_store.graph.get_node(anchor.id)
        reasoning_path_nodes = [anchor.id]
        reasoning_path_names = [anchor_node.name if anchor_node else anchor.id]

        backtrack_stack: List[tuple] = [] if not use_full_ctx else []

        for hop in range(max_iterations):
            current_node = self.graph_store.graph.get_node(current_node_id)
            current_node_name = current_node.name if current_node else current_node_id
            anchor_explored.add(current_node_id)

            hop_decision = decide_fn(
                current_node_id=current_node_id,
                query=query,
                visited_nodes=anchor_explored,
                accumulated_tokens=accumulated_tokens,
                token_limit=effective_token_limit,
                used_chunk_ids=anchor_chunk_ids,
                anchor_knowledge=anchor_knowledge,
                query_embedding=query_embedding,
            )

            this_call_tokens = hop_decision.get("this_call_tokens", 0)
            accumulated_tokens = hop_decision.get("accumulated_tokens", accumulated_tokens)
            anchor_chunk_ids.update(hop_decision.get("new_chunk_ids", []))

            if hop_decision.get("token_limit_reached", False):
                token_limit_reached = True
                judgment = hop_decision
                _safe_print(f"[Agent] 锚点预算达到限制，停止探索", verbose=True)
                break

            _safe_print(f"[Agent] 第{hop+1}跳: [{current_node_id[:30]}] (累计: {accumulated_tokens}/{effective_token_limit})", verbose=True)

            is_sufficient = hop_decision.get("sufficient", False)
            skip_anchor = hop_decision.get("skip_anchor", False) if use_full_ctx else False
            next_nodes = hop_decision.get("next_nodes", [])
            reason = hop_decision.get("reason", "")
            final_answer = hop_decision.get("answer", "")

            _safe_print(f"[Agent] 决策: sufficient={is_sufficient}, next={next_nodes}", verbose=True)

            trace_entry = {
                "anchor_idx": anchor_idx, "anchor_id": anchor.id,
                "hop": hop + 1, "current_node": current_node_id,
                "knowledge_count": len(anchor_knowledge), "decision": hop_decision,
            }
            if use_full_ctx:
                trace_entry["exploration_strategy"] = "full_context"
            reasoning_trace.append(trace_entry)

            if is_sufficient:
                _safe_print(f"[Agent] 找到答案，停止探索", verbose=True)
                iteration_count = hop + 1
                judgment = {"sufficient": True, "answer": final_answer}
                break

            if use_full_ctx and skip_anchor:
                _safe_print(f"[Agent] 跳过锚点 - 该分支无法回答", verbose=True)
                iteration_count = hop + 1
                judgment = {"sufficient": False, "skip_anchor": True}
                break

            if not next_nodes:
                if not use_full_ctx:
                    next_nodes = self._backtrack_or_fallback(
                        current_node_id, anchor_explored, backtrack_stack)
                if not next_nodes:
                    _safe_print(f"[Agent] 没有更多可探索的节点", verbose=True)
                    iteration_count = hop + 1
                    judgment = {"sufficient": False}
                    break

            next_node_id = self._pick_unvisited(next_nodes, anchor_explored)
            if next_node_id is None:
                _safe_print(f"[Agent] 所有候选节点已访问过", verbose=True)
                iteration_count = hop + 1
                judgment = {"sufficient": False}
                break

            if not use_full_ctx:
                self._push_remaining_to_backtrack(
                    current_node_id, next_nodes, anchor_explored, backtrack_stack)

            next_node = self.graph_store.graph.get_node(next_node_id)
            if next_node:
                reasoning_path_nodes.append(next_node_id)
                reasoning_path_names.append(next_node.name)

            _safe_print(f"[Agent] 跳转到 [{next_node_id[:30]}]，已收集 {len(anchor_knowledge)} 条知识", verbose=True)
            current_node_id = next_node_id
            judgment = {"sufficient": False}
        else:
            iteration_count = max_iterations

        if reasoning_path_nodes:
            all_reasoning_paths.append(ReasoningPath(
                anchor_node_id=anchor.id, anchor_score=anchor.score,
                path_nodes=reasoning_path_names, path_node_ids=reasoning_path_nodes,
                path_edges=["→"] * (len(reasoning_path_nodes) - 1),
                path_score=anchor.score,
            ))

        return {
            "anchor_knowledge": anchor_knowledge,
            "anchor_chunk_ids": anchor_chunk_ids,
            "anchor_explored": anchor_explored,
            "judgment": judgment,
            "accumulated_tokens": accumulated_tokens,
            "token_limit_reached": token_limit_reached,
            "iteration_count": iteration_count,
        }

    # -- 保持原有 API 的薄封装 ---
    def _explore_anchor_step_by_step(self, anchor, anchor_idx, query, max_iterations,
                                      accumulated_tokens, reasoning_trace, all_reasoning_paths,
                                      query_embedding=None, anchor_token_limit=None,
                                      collected_chunk_ids=None) -> Dict[str, Any]:
        return self._explore_anchor(
            anchor, anchor_idx, query, max_iterations, accumulated_tokens,
            reasoning_trace, all_reasoning_paths, query_embedding=query_embedding,
            anchor_token_limit=anchor_token_limit, collected_chunk_ids=collected_chunk_ids,
            strategy="step_by_step",
        )

    def _explore_anchor_with_full_context(self, anchor, anchor_idx, query, max_iterations,
                                           accumulated_tokens, reasoning_trace, all_reasoning_paths,
                                           query_embedding=None, anchor_token_limit=None,
                                           collected_chunk_ids=None) -> Dict[str, Any]:
        return self._explore_anchor(
            anchor, anchor_idx, query, max_iterations, accumulated_tokens,
            reasoning_trace, all_reasoning_paths, query_embedding=query_embedding,
            anchor_token_limit=anchor_token_limit, collected_chunk_ids=collected_chunk_ids,
            strategy="full_context",
        )

    # -- step_by_step 回退辅助方法 --

    def _backtrack_or_fallback(self, current_node_id: str, explored: set,
                               backtrack_stack: List[tuple]) -> List[str]:
        """优先尝试未访问的后继，否则从回退栈弹出"""
        unvisited = [s for s in self._get_successors(current_node_id)
                     if s['id'] not in explored]
        if unvisited:
            _safe_print(f"[Agent] 自动选择后继: {unvisited[0]['id'][:30]}...", verbose=True)
            if len(unvisited) > 1:
                backtrack_stack.append((current_node_id, [s['id'] for s in unvisited[1:]]))
            return [unvisited[0]['id']]

        while backtrack_stack:
            prev_id, remaining = backtrack_stack.pop()
            remaining = [b for b in remaining if b not in explored]
            if remaining:
                _safe_print(f"[Agent] 回退到 [{prev_id[:20]}...]，选择分支: {remaining[0][:30]}...", verbose=True)
                if len(remaining) > 1:
                    backtrack_stack.append((prev_id, remaining[1:]))
                return [remaining[0]]
        return []

    @staticmethod
    def _pick_unvisited(candidates: List[str], explored: set) -> Optional[str]:
        for c in candidates:
            if c not in explored:
                return c
        return None

    def _push_remaining_to_backtrack(self, current_node_id: str, selected: List[str],
                                      explored: set, backtrack_stack: List[tuple]):
        all_succ = [s['id'] for s in self._get_successors(current_node_id)
                    if s['id'] not in explored]
        selected_set = set(selected)
        remaining = [s for s in all_succ if s not in selected_set]
        if remaining:
            backtrack_stack.append((current_node_id, remaining))
   

    def _extract_core_event(self, query: str) -> Dict[str, Any]:
        """从问题中提取系统/设备名称，用于锚点检索关键词微调"""
        default_result = {"core_event": query, "system_device": ""}
        
        if not self._llm_client:
            return default_result
        
        prompt = (
            f"从用户问题中提取系统或设备名称（如有），以JSON格式返回。\n\n"
            f"用户问题: {query}\n\n"
            f'输出格式: {{"system_device": "..."}}'
        )

        try:
            from src.utils.openai_client import call_with_retry
            response = call_with_retry(
                lambda: self._llm_client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}, "enable_thinking": False}
                ),
                caller="extract_core_event"
            )
            content = _safe_get_content(response)
            start = content.find('{')
            if start != -1:
                depth, end = 0, -1
                for ci in range(start, len(content)):
                    if content[ci] == '{': depth += 1
                    elif content[ci] == '}':
                        depth -= 1
                        if depth == 0: end = ci + 1; break
                if end != -1:
                    result = json.loads(content[start:end])
                    return {
                        "core_event": query,
                        "system_device": result.get("system_device", ""),
                    }
            
        except Exception as e:
            _safe_print(f"[设备提取] 提取失败: {e}")
        
        return default_result
    
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """余弦相似度（安全处理零向量和 NaN）"""
        vec1 = np.asarray(vec1, dtype=np.float64).flatten()
        vec2 = np.asarray(vec2, dtype=np.float64).flatten()
        if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
   
    
    def close(self):
        self.graph_store.close()
