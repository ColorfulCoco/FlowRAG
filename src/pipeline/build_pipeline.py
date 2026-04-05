# -*- coding: utf-8 -*-
"""
构建阶段流水线

整合图谱构建、文档处理、节点嵌入和知识挂载的完整构建流程。

Author: CongCongTian
"""

import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

from src.models.schemas import KnowledgeGraph, TextChunk
from src.embedding.embedder import Embedder
from src.embedding.context_embedder import ContextEmbedder
from src.vector_store.vector_db import VectorDB
from src.graph.graph_store import GraphStore
from src.graph.mermaid_parser import (
    MermaidParser,
    GraphGenerator,
    MermaidOnlyGenerator,
)
from src.utils.text_splitter import TextSplitter, SemanticTextSplitter, SentenceSplitter, ParagraphSplitter
from src.knowledge.keyword_extractor_local import LocalKeywordExtractor
from src.knowledge.keyword_index import KeywordIndex
from src.knowledge.knowledge_mounter import KnowledgeMounter
from src.pipeline.build_cache import BuildCache
from src.utils.token_counter import TokenStatistics
from src.config import (
    BUILD_LLM_API_KEY, BUILD_LLM_BASE_URL, BUILD_LLM_MODEL,
    BUILD_EMB_API_KEY, BUILD_EMB_BASE_URL, BUILD_EMB_MODEL,
)


class BuildPipeline:
    """构建阶段流水线
    
    流程: 图谱构建 -> 文档分块入库 -> 关键词提取 -> 节点嵌入 -> 知识挂载 -> 持久化
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        emb_api_key: Optional[str] = None,
        emb_base_url: Optional[str] = None,
        chat_model_name: Optional[str] = None,
        chat_model_base_url: Optional[str] = None,
        two_stage_mode: bool = False,
        two_stage_model_name: Optional[str] = None,
        two_stage_diagram_type: str = "both",
        two_stage_merge_strategy: str = "flowchart_priority",
        use_llm_mermaid_parse: bool = False,
        persist_directory: str = "./data/flowrag_store",
        vector_backend: str = "faiss",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_semantic_split: bool = True,
        split_by_sentence: bool = False,
        split_by_paragraph: bool = False,
        max_sentences_per_chunk: int = 2,
        max_enrichment_chunks: int = 3,
    ):
        """初始化构建流水线
        
        Args:
            two_stage_mode: 两阶段模式 (TXT->Mermaid->Graph)，False 时使用微调模型一步式输出
            two_stage_diagram_type: 图表类型 "flowchart"/"sequence"/"both"
            two_stage_merge_strategy: 两图合并策略 "flowchart_priority"/"sequence_priority"/"merge"
            use_llm_mermaid_parse: True 用大模型解析 Mermaid，False 用规则解析（更快）
            split_by_paragraph: 纯段落分块（优先级最高）
            split_by_sentence: 按句子分块
        """
        api_key = api_key or BUILD_LLM_API_KEY
        base_url = base_url or BUILD_LLM_BASE_URL
        embedding_model = embedding_model or BUILD_EMB_MODEL
        emb_api_key = emb_api_key or BUILD_EMB_API_KEY
        emb_base_url = emb_base_url or BUILD_EMB_BASE_URL
        
        self.two_stage_mode = two_stage_mode
        self.two_stage_diagram_type = two_stage_diagram_type
        self.two_stage_merge_strategy = two_stage_merge_strategy
        self.use_llm_mermaid_parse = use_llm_mermaid_parse
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key
        self.base_url = base_url
        
        self.token_stats = TokenStatistics()
        
        self.embedder = Embedder(
            embedder_type="openai",
            api_key=emb_api_key,
            base_url=emb_base_url,
            model_name=embedding_model,
            token_stats=self.token_stats,
            embedding_type="text",
        )
        
        # 一步式生成器：微调模型直接输出 Graph JSON
        self.graph_generator = None
        if chat_model_name and api_key:
            self.graph_generator = GraphGenerator(
                api_key=api_key,
                base_url=chat_model_base_url or base_url or "",
                model_name=chat_model_name,
                token_stats=self.token_stats,
            )
        
        # 两阶段生成器：先 Mermaid 再解析为图谱
        self.mermaid_only_generator = None
        if two_stage_mode and api_key:
            mermaid_model = two_stage_model_name or BUILD_LLM_MODEL
            self.mermaid_only_generator = MermaidOnlyGenerator(
                api_key=api_key,
                base_url=base_url or BUILD_LLM_BASE_URL,
                model_name=mermaid_model,
                token_stats=self.token_stats,
            )
        
        self.mermaid_parser = MermaidParser(
            api_key=api_key or "",
            base_url=base_url or BUILD_LLM_BASE_URL,
            model_name=BUILD_LLM_MODEL,
            use_llm_parse=use_llm_mermaid_parse,
            token_stats=self.token_stats,
        )
        
        self.mermaid_output_dir = self.persist_directory / "mermaid_visualization"
        self.mermaid_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_file = str(self.persist_directory / "node_embeddings.npz")
        
        self.graph_store = GraphStore(
            namespace="flow_rag",
            persist_directory=str(self.persist_directory),
        )
        
        self.vector_db = VectorDB(
            backend=vector_backend,
            index_type="Flat",
            dimension=self.embedder.dimension,
            persist_directory=str(self.persist_directory / "vector_store"),
        )
        
        # 分块器优先级: paragraph > sentence > semantic > basic
        if split_by_paragraph:
            self.text_splitter = ParagraphSplitter(min_chunk_length=10)
        elif split_by_sentence:
            self.text_splitter = SentenceSplitter(
                max_sentences_per_chunk=max_sentences_per_chunk,
                min_chunk_length=20,
                max_chunk_length=300,
            )
        elif use_semantic_split:
            self.text_splitter = SemanticTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            self.text_splitter = TextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        
        self.context_embedder = ContextEmbedder(self.embedder)
        
        self.keyword_extractor = LocalKeywordExtractor(
            top_k=5,
            embedder=self.embedder,
            use_mmr=True,
        )
        
        self.keyword_index = KeywordIndex(
            embedder=self.embedder,
            persist_path=str(self.persist_directory / "keyword_index"),
        )
        
        self.knowledge_mounter = KnowledgeMounter(
            vector_db=self.vector_db,
            top_k_chunks=max_enrichment_chunks,
        )
        
        cache_file = self.persist_directory / "build_cache.json"
        self.build_cache = BuildCache(cache_file=str(cache_file))
        
        # 本次处理的文件记录（阶段完成后 flush 到 build_cache）
        self.processed_file_records = []
        
        cumulative_tokens = self.build_cache.get_cumulative_tokens()
        if cumulative_tokens:
            self.token_stats.cumulative_graph_generation_tokens = cumulative_tokens.get("graph_generation", 0)
            self.token_stats.cumulative_graph_generation_prompt_tokens = cumulative_tokens.get("graph_generation_prompt", 0)
            self.token_stats.cumulative_graph_generation_completion_tokens = cumulative_tokens.get("graph_generation_completion", 0)
            self.token_stats.cumulative_text_embedding_tokens = cumulative_tokens.get("text_embedding", 0)
            self.token_stats.cumulative_keyword_embedding_tokens = cumulative_tokens.get("keyword_embedding", 0)
            self.token_stats.cumulative_node_embedding_tokens = cumulative_tokens.get("node_embedding", 0)
        
        self.stats = {}
        
        # chunk 文本 -> embedding 缓存，跨阶段复用（process_documents 填充，extract_keywords MMR 复用）
        self._chunk_text_embeddings = {}  # Dict[str, numpy.ndarray]
    
    def _build_graph_core(
        self,
        text_folder: str,
        pattern: str,
        force_rebuild: bool,
        enable_parallel: bool,
        max_workers: int,
        process_file_fn,
        mode_label: str = "",
        extra_stats: Optional[Dict] = None,
    ) -> KnowledgeGraph:
        """图谱构建统一入口，由 process_file_fn(fp, tg_id, text, lock) 提供单文件处理逻辑"""
        from src.config import LOG_VERBOSE
        if force_rebuild:
            if LOG_VERBOSE:
                print(f"      [缓存] 强制重建，清除所有缓存")
            self.build_cache.clear()

        folder = Path(text_folder)
        files = sorted(folder.glob(pattern))
        if not files:
            raise ValueError(f"No files matching pattern: {pattern}")

        files_to_process, cached_files = [], []
        for fp in files:
            (files_to_process if (force_rebuild or not self.build_cache.is_file_cached(fp)) else cached_files).append(fp)

        if LOG_VERBOSE:
            print(f"      [缓存] 总数={len(files)} 已缓存={len(cached_files)} 待处理={len(files_to_process)}")
            if cached_files:
                for fp in cached_files:
                    rec = self.build_cache.get_file_record(fp.name)
                    if rec:
                        print(f"        跳过 {fp.name} (节点={rec.nodes_count} 边={rec.edges_count})")

        mermaid_results: List[Dict] = []
        generation_log: List[Dict] = []

        if cached_files and not force_rebuild:
            merged_graph = self._load_cached_graph()
            self._load_cached_mermaid(cached_files, mermaid_results)
        else:
            merged_graph = KnowledgeGraph()

        if not files_to_process:
            if LOG_VERBOSE:
                print(f"      所有文件已缓存，无需处理")
        elif LOG_VERBOSE:
            print(f"      处理 {len(files_to_process)} 个文件...")

        total = len(files_to_process)
        progress_lock = threading.Lock()

        def _wrapped(fp, pbar=None):
            tg_id = fp.stem
            if pbar:
                with progress_lock:
                    pbar.set_description(f"{tg_id}")
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()

            graph, log_entry, mermaid_entry = process_file_fn(fp, tg_id, text, progress_lock)

            nodes_count = len(graph.nodes) if graph else 0
            edges_count = len(graph.edges) if graph else 0
            with progress_lock:
                self.processed_file_records.append({
                    "file_path": fp, "tg_id": tg_id,
                    "nodes_count": nodes_count, "edges_count": edges_count,
                    "success": graph is not None and nodes_count > 0,
                    "error_message": log_entry.get("error"),
                })
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({"nodes": nodes_count, "edges": edges_count})
            return graph, log_entry, mermaid_entry

        desc = f"构建图谱{' (' + mode_label + ')' if mode_label else ''}"
        if enable_parallel and total > 1:
            with tqdm(total=total, desc=desc, unit="file", ncols=100) as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    fmap = {executor.submit(_wrapped, fp, pbar): fp for fp in files_to_process}
                    for fut in as_completed(fmap):
                        try:
                            g, log, mmd = fut.result()
                            if g and g.nodes:
                                with progress_lock:
                                    self._merge_graphs(merged_graph, g)
                            generation_log.append(log)
                            if mmd:
                                mermaid_results.append(mmd)
                        except Exception as e:
                            fp = fmap[fut]
                            pbar.write(f"      [错误] {fp.name}: {e}")
                            generation_log.append({"tg_id": fp.stem, "success": False, "error": str(e)})
        else:
            with tqdm(total=total, desc=desc, unit="file", ncols=100) as pbar:
                for fp in files_to_process:
                    g, log, mmd = _wrapped(fp, pbar)
                    if g and g.nodes:
                        self._merge_graphs(merged_graph, g)
                    generation_log.append(log)
                    if mmd:
                        mermaid_results.append(mmd)

        self._save_build_artifacts(merged_graph, generation_log, mermaid_results)

        # 图谱阶段完成后立即 flush，即使后续阶段中断也能保留进度
        if self.processed_file_records:
            for record in self.processed_file_records:
                self.build_cache.add_file_record(
                    file_path=record["file_path"],
                    tg_id=record["tg_id"],
                    nodes_count=record["nodes_count"],
                    edges_count=record["edges_count"],
                    success=record["success"],
                    error_message=record.get("error_message"),
                )
            self.build_cache.save()
            if LOG_VERBOSE:
                print(f"      [缓存] 已保存 {len(self.processed_file_records)} 条文件记录")
            self.processed_file_records.clear()

        self.stats["graph"] = {
            "nodes": len(merged_graph.nodes), "edges": len(merged_graph.edges),
            "source_files": len(files), "processed_files": len(files_to_process),
            "cached_files": len(cached_files), **(extra_stats or {}),
        }
        return merged_graph

    def _load_cached_graph(self) -> KnowledgeGraph:
        gpath = self.persist_directory / "knowledge_graph.json"
        if gpath.exists():
            try:
                with open(gpath, "r", encoding="utf-8") as f:
                    return KnowledgeGraph.from_dict(json.load(f))
            except Exception as e:
                print(f"      [缓存] 加载失败: {e}，从空图谱开始")
        return KnowledgeGraph()

    def _load_cached_mermaid(self, cached_files, out: list):
        for fp in cached_files:
            tg_id = fp.stem
            entry: Dict[str, Any] = {}
            for suffix in ("_flowchart.mmd", "_sequence.mmd", "_mermaid.mmd"):
                mf = self.mermaid_output_dir / f"{tg_id}{suffix}"
                if mf.exists():
                    key = suffix.replace("_", "").replace(".mmd", "")
                    with open(mf, "r", encoding="utf-8") as f:
                        entry[key] = f.read()
            if entry:
                entry["tg_id"] = tg_id
                out.append(entry)

    def _save_mermaid_codes(self, tg_id, mermaid_codes, text, progress_lock) -> Optional[Dict]:
        """将 Mermaid 代码写入磁盘并返回 mermaid_entry 字典（无内容时返回 None）"""
        from src.config import LOG_VERBOSE
        has_fc = getattr(mermaid_codes, "flowchart", None)
        has_sq = getattr(mermaid_codes, "sequence", None)
        if not has_fc and not has_sq:
            return None
        self.mermaid_output_dir.mkdir(parents=True, exist_ok=True)
        entry: Dict[str, Any] = {"tg_id": tg_id, "title": (text.split("\n")[0] if text else tg_id)}
        saved = []
        if has_fc:
            entry["flowchart"] = mermaid_codes.flowchart
            (self.mermaid_output_dir / f"{tg_id}_flowchart.mmd").write_text(mermaid_codes.flowchart, encoding="utf-8")
            saved.append("flowchart")
        if has_sq:
            entry["sequence"] = mermaid_codes.sequence
            (self.mermaid_output_dir / f"{tg_id}_sequence.mmd").write_text(mermaid_codes.sequence, encoding="utf-8")
            saved.append("sequence")
        if saved and progress_lock:
            with progress_lock:
                    if LOG_VERBOSE:
                        print(f"        mermaid 已保存: {', '.join(saved)}")
        return entry

    def _save_build_artifacts(self, graph, log, mermaid):
        from src.config import LOG_VERBOSE
        if mermaid:
            p = self.persist_directory / "mermaid_visualization.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(mermaid, f, ensure_ascii=False, indent=2)
        with open(self.persist_directory / "generation_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        gp = self.persist_directory / "knowledge_graph.json"
        with open(gp, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        if LOG_VERBOSE:
            print(f"      图谱已保存: {gp}")
            print(f"      正在保存图谱到本地存储...")
        self.graph_store.save_graph(graph, clear_existing=True)

    # ------------------------------------------------------------------
    # 三种构建模式，均委托 _build_graph_core
    # ------------------------------------------------------------------

    def build_graph_from_texts(self, text_folder, pattern="TG-*.txt",
                               force_rebuild=False, enable_parallel=True, max_workers=4):
        """一步式构建：微调模型直接输出 Graph JSON"""
        from src.config import LOG_VERBOSE
        if not self.graph_generator:
            raise ValueError("graph_generator not configured (set chat_model_name)")
        if LOG_VERBOSE:
            print(f"[1/6] one-stage graph build: {text_folder}")

        def _process(fp, tg_id, text, lock):
            result = self.graph_generator.generate(text, tg_id)
            log = {"tg_id": tg_id, "success": result.success,
                   "error": result.error_message if not result.success else None}
            mmd = None
            if result.success and result.graph:
                mmd = self._save_mermaid_codes(tg_id, result.mermaid_codes, text, lock)
                log.update(nodes=len(result.graph.nodes), edges=len(result.graph.edges))
            else:
                with lock:
                    print(f"        [warn] {tg_id} failed: {result.error_message}")
            return result.graph if result.success else None, log, mmd

        return self._build_graph_core(text_folder, pattern, force_rebuild,
                                      enable_parallel, max_workers, _process, "one-stage")

    def build_graph_from_texts_two_stage(self, text_folder, pattern="TG-*.txt",
                                         force_rebuild=False, enable_parallel=True, max_workers=4):
        """两阶段构建：LLM -> Mermaid -> MermaidParser -> Graph"""
        from src.config import LOG_VERBOSE
        if not self.mermaid_only_generator:
            raise ValueError("mermaid_only_generator not configured (set two_stage_mode=True)")
        if LOG_VERBOSE:
            print(f"[1/6] two-stage graph build: {text_folder}")

        def _process(fp, tg_id, text, lock):
            mresult = self.mermaid_only_generator.generate(text, tg_id)
            log: Dict[str, Any] = {"tg_id": tg_id, "mode": "two_stage",
                                   "stage1_success": mresult.success,
                                   "error": mresult.error_message if not mresult.success else None}
            graph, mmd = None, None
            if mresult.success and mresult.mermaid_codes.has_any():
                mmd = self._save_mermaid_codes(tg_id, mresult.mermaid_codes, text, lock)
                graph = self.mermaid_parser.mermaid_codes_to_knowledge_graph(
                    mermaid_codes=mresult.mermaid_codes, source_tg=tg_id,
                    tg_title=text.split("\n")[0] if text else tg_id, keywords=mresult.keywords)
                if graph.nodes:
                    log.update(stage2_success=True, nodes=len(graph.nodes), edges=len(graph.edges))
                else:
                    graph = None
                    log.update(stage2_success=False, error="empty graph after mermaid parse")
            else:
                with lock:
                    print(f"        [warn] {tg_id} mermaid failed: {mresult.error_message}")
            return graph, log, mmd

        return self._build_graph_core(text_folder, pattern, force_rebuild,
                                      enable_parallel, max_workers, _process, "two-stage",
                                      {"mode": "two_stage", "diagram_type": self.two_stage_diagram_type})

    def build_graph_from_saved_json(self, json_path: str) -> KnowledgeGraph:
        """从已保存的图谱 JSON 文件加载（用于恢复/调试）"""
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"[1/6] 从保存的图谱JSON加载: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        graph = KnowledgeGraph.from_dict(data)
        
        if LOG_VERBOSE:
            print(f"      正在保存图谱到本地存储...")
        self.graph_store.save_graph(graph, clear_existing=True)
        
        self.stats["graph"] = {"nodes": len(graph.nodes), "edges": len(graph.edges)}
        if LOG_VERBOSE:
            print(f"      已加载 {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
        
        return graph
    
    def load_existing_graph(self) -> KnowledgeGraph:
        """从本地文件加载已有图谱"""
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"[1/6] 从本地文件加载图谱...")
        graph = self.graph_store.load_graph()
        self.stats["graph"] = {"nodes": len(graph.nodes), "edges": len(graph.edges)}
        if LOG_VERBOSE:
            print(f"      已加载 {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
        return graph
    
    def _merge_graphs(self, target: KnowledgeGraph, source: KnowledgeGraph):
        """合并图谱（同 ID 节点合并 source_tg 信息）"""
        for node_id, node in source.nodes.items():
            if node_id in target.nodes:
                existing = target.nodes[node_id]
                for tg in node.source_tg:
                    if tg not in existing.source_tg:
                        existing.source_tg.append(tg)
            else:
                target.nodes[node_id] = node
        
        existing_edges = {(e.source_id, e.target_id) for e in target.edges}
        for edge in source.edges:
            if (edge.source_id, edge.target_id) not in existing_edges:
                target.edges.append(edge)
                existing_edges.add((edge.source_id, edge.target_id))
        
        target._rebuild_edge_index()

        for node in target.nodes.values():
            node.predecessors.clear()
            node.successors.clear()
        for edge in target.edges:
            if edge.source_id in target.nodes:
                if edge.target_id not in target.nodes[edge.source_id].successors:
                    target.nodes[edge.source_id].successors.append(edge.target_id)
            if edge.target_id in target.nodes:
                if edge.source_id not in target.nodes[edge.target_id].predecessors:
                    target.nodes[edge.target_id].predecessors.append(edge.source_id)
    
    def process_documents(self, folder_path: str, pattern: str = "*.txt", force_rebuild: bool = False) -> List[TextChunk]:
        """处理陈述性文档并存入向量数据库（文件级增量构建）"""
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"[2/6] 处理陈述性文档: {folder_path}")
        folder = Path(folder_path)
        
        all_files = list(folder.glob(pattern))
        if not all_files:
            print("      [警告] 没有文档可索引")
            return []
        
        if force_rebuild:
            files_to_process = all_files
            if LOG_VERBOSE:
                print(f"      [强制重建] 将处理所有 {len(files_to_process)} 个文件")
        else:
            files_to_process = [
                f for f in all_files 
                if not self.build_cache.is_file_cached(f)
            ]
            
            if not files_to_process:
                if LOG_VERBOSE:
                    print(f"      [缓存] 所有文件都已处理，跳过文档处理")
                chunks = self._load_chunks_from_vector_db()
                if chunks:
                    if LOG_VERBOSE:
                        print(f"      [缓存] 已加载 {len(chunks)} 个文本块（0 token）")
                    self.stats["documents"] = {
                        "total_chunks": len(chunks),
                        "source_files": len(set(c.source_file for c in chunks)),
                        "from_cache": True,
                    }
                    return chunks
                else:
                    if LOG_VERBOSE:
                        print(f"      [缓存] 加载失败，将重新处理所有文件")
                    files_to_process = all_files
            else:
                if LOG_VERBOSE:
                    print(f"      [增量构建] 检测到 {len(files_to_process)} 个新文件或修改文件")
                    print(f"      [缓存] 跳过 {len(all_files) - len(files_to_process)} 个已处理文件")
        
        new_chunks = []
        for file_path in files_to_process:
            file_chunks = self.text_splitter.split_file(str(file_path))
            new_chunks.extend(file_chunks)
        
        if not new_chunks:
            print("      警告：新文件为空或无法分块")
            return self._load_chunks_from_vector_db()
        
        if LOG_VERBOSE:
            print(f"      新处理 {len(new_chunks)} 个文本块（来自 {len(files_to_process)} 个文件）")
        
        if LOG_VERBOSE:
            print(f"[3/6] 生成文档嵌入...")
        texts = [chunk.text for chunk in new_chunks]
        
        self.embedder.set_embedding_type("text")
        embeddings = self.embedder.embed(texts)
        
        if LOG_VERBOSE:
            if len(files_to_process) < len(all_files):
                print(f"      已生成嵌入 (新文件: {len(files_to_process)} 个)")
            else:
                print(f"      已生成嵌入")
        
        # 缓存 chunk embedding，供 extract_keywords MMR 复用（键与 MMR 一致取 text[:1000]）
        for chunk, emb in zip(new_chunks, embeddings):
            cache_key = chunk.text[:1000]
            self._chunk_text_embeddings[cache_key] = emb
        
        self.vector_db.add(new_chunks, embeddings)
        if LOG_VERBOSE:
            print(f"      已存入向量数据库")
        
        all_chunks = self._load_chunks_from_vector_db()
        
        self.stats["documents"] = {
            "total_chunks": len(all_chunks),
            "source_files": len(all_files),
            "from_cache": False,
            "new_files": len(files_to_process),
            "cached_files": len(all_files) - len(files_to_process),
        }
        
        self.build_cache.update_stage2_cache(
            folder_path=folder_path,
            pattern=pattern,
            documents_processed=True,
        )
        
        return all_chunks
    
    def _load_chunks_from_vector_db(self) -> List[TextChunk]:
        """从向量数据库内存中获取所有 chunks（不涉及磁盘读取）"""
        return self.vector_db.get_all_chunks()
    
    def extract_keywords(self, chunks: List[TextChunk], show_progress: bool = True, force_rebuild: bool = False):
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"[关键词提取] 从 {len(chunks)} 个知识块中提取关键词（jieba.textrank，0 token）...")
        
        if not chunks:
            print("      警告：没有知识块需要提取关键词")
            return
        

        if not force_rebuild:
            try:
                self.keyword_index._load()
            except Exception:
                pass
        
        if force_rebuild:
            chunks_to_process = chunks
            if LOG_VERBOSE:
                print(f"      [强制重建] 将处理所有 {len(chunks_to_process)} 个知识块")
        else:
            chunks_to_process = [
                chunk for chunk in chunks
                if chunk.id not in self.keyword_index._chunk_to_keywords
            ]
            
            if not chunks_to_process:
                if LOG_VERBOSE:
                    print(f"      [缓存] 所有知识块都已提取关键词，跳过")
                keyword_stats = self.keyword_index.get_statistics()
                if LOG_VERBOSE:
                    print(f"      [缓存] 已加载 {keyword_stats['total_keywords']} 个关键词（0 token）")
                
                self.stats["keywords"] = {
                    **keyword_stats,
                    "extraction_stats": {
                        "method": "从缓存加载",
                        "token_cost": 0,
                    },
                }
                return
            else:
                if LOG_VERBOSE:
                    print(f"      [增量构建] 检测到 {len(chunks_to_process)} 个新知识块需要提取关键词")

        
        # 预填充 embedding 缓存，复用 process_documents 阶段的结果（键统一为 text[:1000]）
        if self._chunk_text_embeddings:
            pre_cached = 0
            for chunk in chunks_to_process:
                cache_key = chunk.text[:1000]
                if cache_key in self._chunk_text_embeddings:
                    self.keyword_extractor._embedding_cache[cache_key] = self._chunk_text_embeddings[cache_key]
                    pre_cached += 1
            if pre_cached > 0 and LOG_VERBOSE:
                print(f"      [优化] 复用 {pre_cached} 个 chunk 的已有 embedding，避免 MMR 重复调用 API")
        
        texts = [chunk.text for chunk in chunks_to_process]
        results = self.keyword_extractor.extract_batch(texts, show_progress=show_progress)
        
        new_keywords = set()
        new_keyword_links = 0
        
        for chunk, result in zip(chunks_to_process, results):
            if result.keywords:
                self.keyword_index.add_chunk_keywords(chunk.id, result.keywords)
                new_keyword_links += len(result.keywords)
                new_keywords.update(result.keywords)
            else:
                if chunk.id not in self.keyword_index._chunk_to_keywords:
                    self.keyword_index._chunk_to_keywords[chunk.id] = []
        
        if LOG_VERBOSE:
            print(f"      新提取了 {new_keyword_links} 个关键词关联（来自 {len(chunks_to_process)} 个知识块）")
        
        # 复用 MMR 过程中已生成的候选词 embedding，避免重复 API 调用
        pre_computed = {}
        if hasattr(self.keyword_extractor, '_embedding_cache'):
            for kw in new_keywords:
                if kw in self.keyword_extractor._embedding_cache:
                    pre_computed[kw] = self.keyword_extractor._embedding_cache[kw]
            if pre_computed and LOG_VERBOSE:
                print(f"      [优化] 复用 {len(pre_computed)}/{len(new_keywords)} 个关键词的 MMR 已有嵌入")
        
        self.embedder.set_embedding_type("keyword")
        self.keyword_index.build_embeddings(
            show_progress=show_progress, 
            incremental=True,
            pre_computed_embeddings=pre_computed,
        )
        
        self.keyword_index._save()
        
        keyword_stats = self.keyword_index.get_statistics()
        extractor_stats = self.keyword_extractor.get_statistics()
        self.stats["keywords"] = {
            **keyword_stats,
            "extraction_stats": extractor_stats,
        }
        if LOG_VERBOSE:
            print(f"      关键词索引: {keyword_stats['total_keywords']} 个关键词")
        
        self.build_cache.update_stage2_cache(
            keywords_extracted=True,
        )
    
    def embed_graph_nodes(
        self,
        method: str = "residual",
        gamma: float = 0.5,
        max_hops: int = 2,
        force_rebuild: bool = False,
        folder_path: Optional[str] = None,
        pattern: str = "*.txt",
    ):
        """为图谱节点生成上下文感知嵌入（支持缓存）
        
        Args:
            method: "residual"(默认) / "plain" / "weighted"
            gamma: 残差衰减因子
            max_hops: 最大跳数 K
        """
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"[4/6] 生成图谱节点嵌入 (方法: {method}, γ={gamma}, K={max_hops})...")
        
        graph = self.graph_store.graph
        
        if folder_path and not force_rebuild and self.build_cache.should_skip_embeddings_generation(folder_path, pattern):
            if LOG_VERBOSE:
                print(f"      [缓存] 所有文件都已处理且已生成嵌入，从已保存的嵌入文件加载...")
            embeddings = self._load_cached_embeddings()
            if embeddings is not None:
                if len(embeddings) != len(graph.nodes):
                    if LOG_VERBOSE:
                        print(f"      [缓存] 嵌入文件中有 {len(embeddings)} 个节点，但图谱有 {len(graph.nodes)} 个节点，需要重新生成")
                else:
                    self.graph_store.set_node_embeddings(embeddings)
                    if LOG_VERBOSE:
                        print(f"      [缓存] 已加载 {len(embeddings)} 个节点的嵌入")
                    
                    self.stats["node_embeddings"] = len(embeddings)
                    self.stats["embedding_method"] = method
                    self.stats["embedding_gamma"] = gamma
                    self.stats["embedding_max_hops"] = max_hops
                    return
            else:
                if LOG_VERBOSE:
                    print(f"      [缓存] 嵌入文件不存在或加载失败，将重新生成")
        
        cached_files = set(self.build_cache.get_cached_files())
        new_node_ids = set()
        
        for node_id, node in graph.nodes.items():
            for tg in node.source_tg:
                if f"{tg}.txt" not in cached_files and tg not in cached_files:
                    new_node_ids.add(node_id)
                    break
        
        has_embedding_file = Path(self.embedding_file).exists()
        is_first_build = not has_embedding_file
        
        if is_first_build:
            self._embed_all_nodes(graph, method, gamma, max_hops)
        elif new_node_ids:
            self._embed_incremental_nodes(graph, new_node_ids, method, gamma, max_hops)
        else:
            embeddings = self._load_cached_embeddings()
            if embeddings is not None:
                self.graph_store.set_node_embeddings(embeddings)
                if LOG_VERBOSE:
                    print(f"      已加载 {len(embeddings)} 个节点的 embedding")
            else:
                self._embed_all_nodes(graph, method, gamma, max_hops)
        
        current_embeddings = self.graph_store._node_embeddings
        if LOG_VERBOSE:
            print(f"      完成，当前共有 {len(current_embeddings)} 个节点的 embedding")
        self.stats["node_embeddings"] = len(current_embeddings)
        self.stats["embedding_method"] = method
        self.stats["embedding_gamma"] = gamma
        self.stats["embedding_max_hops"] = max_hops
        
        if folder_path:
            self.build_cache.update_stage2_cache(
                folder_path=folder_path,
                pattern=pattern,
                embeddings_generated=True,
            )

    def _load_cached_embeddings(self) -> Optional[Dict[str, Any]]:
        """从 npz 文件加载已保存的节点嵌入"""
        try:
            import numpy as np
            if Path(self.embedding_file).exists():
                with np.load(self.embedding_file, allow_pickle=True) as data:
                    node_ids = data['node_ids']
                    embeddings_matrix = data['embeddings']
                    return {
                        str(nid): emb for nid, emb in zip(node_ids, embeddings_matrix)
                    }
        except Exception as e:
            print(f"      [缓存] 加载失败: {e}")
        return None

    def _embed_all_nodes(self, graph, method: str, gamma: float, max_hops: int):
        """为所有节点生成 embedding"""
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"      [首次构建] 为所有 {len(graph.nodes)} 个节点生成 embedding...")
        self.embedder.set_embedding_type("node")
        embeddings = self.context_embedder.embed_all_nodes(
            graph,
            method=method,
            gamma=gamma,
            max_hops=max_hops,
            show_progress=True,
        )
        self.graph_store.set_node_embeddings(embeddings)
        if LOG_VERBOSE:
            print(f"      已生成节点嵌入 (所有节点: {len(graph.nodes)} 个)")

    def _embed_incremental_nodes(self, graph, new_node_ids: set, method: str, gamma: float, max_hops: int):
        """增量生成新节点的 embedding（旧节点从缓存加载）"""
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"      [增量构建] 检测到 {len(new_node_ids)} 个新节点，只为新节点生成 embedding...")
        
        old_embeddings = self._load_cached_embeddings()
        load_old_failed = False
        if old_embeddings is not None:
            if LOG_VERBOSE:
                print(f"      已加载 {len(old_embeddings)} 个旧节点的 embedding")
        else:
            print(f"      [警告] 加载旧 embedding 失败，将重新生成所有节点")
            old_embeddings = {}
            new_node_ids = set(graph.nodes.keys())
            load_old_failed = True
        
        self.embedder.set_embedding_type("node")
        
        # 新节点及其邻居都需要重新生成基础 embedding
        nodes_need_embedding = set(new_node_ids)
        for node_id in new_node_ids:
            hop_neighbors = self.context_embedder._get_k_hop_neighbors(
                graph.nodes[node_id], graph, max_hops
            )
            for hop, neighbor_ids in hop_neighbors.items():
                nodes_need_embedding.update(neighbor_ids)
        
        ordered_ids = sorted(nid for nid in nodes_need_embedding if nid in graph.nodes)
        nodes_to_embed = [graph.nodes[nid].name for nid in ordered_ids]
        if nodes_to_embed:
            batch_embeddings = self.embedder.embed(nodes_to_embed)
            for i, nid in enumerate(ordered_ids):
                self.context_embedder._node_embedding_cache[nid] = batch_embeddings[i]
        
        import numpy as np
        new_embeddings = {}
        for node_id in new_node_ids:
            node = graph.nodes[node_id]
            if method == "residual":
                embedding = self.context_embedder.embed_node_residual(
                    node, graph, gamma=gamma, max_hops=max_hops)
            elif method == "weighted":
                embedding = self.context_embedder.embed_node_weighted(node, graph)
            else:
                emb = self.context_embedder._get_node_base_embedding(node)
                norm = np.linalg.norm(emb)
                embedding = emb / norm if norm > 0 else emb
            new_embeddings[node_id] = embedding
        
        self.context_embedder._node_embedding_cache.clear()
        
        final_embeddings = {**old_embeddings, **new_embeddings}
        self.graph_store.set_node_embeddings(final_embeddings)
        
        if LOG_VERBOSE:
            if load_old_failed:
                print(f"      已生成节点嵌入 (所有节点: {len(final_embeddings)} 个)")
            else:
                print(f"      已生成节点嵌入 (新节点: {len(new_embeddings)} 个, 合并后共 {len(final_embeddings)} 个)")
    
    def mount_knowledge(self, debug: bool = False, only_new_nodes: bool = False):
        """为图谱节点挂载相似的陈述性知识块"""
        from src.config import LOG_VERBOSE
        if LOG_VERBOSE:
            print(f"[5/6] 知识挂载（节点找知识）...")
        
        graph = self.knowledge_mounter.mount_knowledge(
            graph=self.graph_store.graph,
            show_progress=True,
            debug=debug,
            only_new_nodes=only_new_nodes,
        )
        self.graph_store.save_graph(graph)
        
        graph_json_path = self.persist_directory / "knowledge_graph.json"
        with open(graph_json_path, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        
        mount_stats = self.knowledge_mounter.get_statistics()
        self.stats["knowledge_mounting"] = mount_stats
        
        if LOG_VERBOSE:
            print(f"      挂载完成: {mount_stats['nodes_with_chunks']} 个节点有挂载, 总挂载 {mount_stats['total_mountings']} 次")
    
    def _build_or_load_graph(
        self,
        skip_graph_build: bool,
        saved_graph_json: Optional[str],
        logic_text_folder: Optional[str],
        logic_text_pattern: str,
        force_rebuild: bool,
        max_workers: int,
        use_two_stage_mode: bool,
    ):
        from src.config import LOG_VERBOSE
        if skip_graph_build:
            self.load_existing_graph()
        elif saved_graph_json:
            self.build_graph_from_saved_json(saved_graph_json)
        elif logic_text_folder:
            if use_two_stage_mode:
                if not self.mermaid_only_generator:
                    raise ValueError(
                        "两阶段模式需要配置 two_stage_mode=True 和有效的 api_key。"
                        "如果要使用一步式模式，请设置 use_two_stage=False"
                    )
                if LOG_VERBOSE:
                    print(f"[模式] 使用两阶段模式: TXT → Mermaid → Graph")
                self.build_graph_from_texts_two_stage(logic_text_folder, pattern=logic_text_pattern, force_rebuild=force_rebuild, max_workers=max_workers)
            else:
                if not self.graph_generator:
                    raise ValueError(
                        "一步式模式需要配置对话模型 chat_model_name。"
                        "如果要使用两阶段模式，请设置 use_two_stage=True 或 two_stage_mode=True"
                    )
                if LOG_VERBOSE:
                    print(f"[模式] 使用一步式模式: TXT → 微调模型 → Graph JSON")
                self.build_graph_from_texts(logic_text_folder, pattern=logic_text_pattern, force_rebuild=force_rebuild, max_workers=max_workers)
        else:
            raise ValueError("必须提供 logic_text_folder 或 saved_graph_json，或设置 skip_graph_build=True")

    def _run_phase2(
        self,
        chunks: List[TextChunk],
        documents_folder: Optional[str],
        document_pattern: str,
        logic_text_folder: Optional[str],
        logic_text_pattern: str,
        force_rebuild: bool,
        embedding_method: str,
        embedding_gamma: float,
        embedding_max_hops: int,
        enable_parallel: bool,
        max_workers: int,
    ):
        if enable_parallel and chunks:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                
                if chunks and documents_folder:
                    def extract_keywords_task():
                        self.extract_keywords(chunks, force_rebuild=force_rebuild)
                    
                    futures['keywords'] = executor.submit(extract_keywords_task)
                
                def embed_nodes_task():
                    self.embed_graph_nodes(
                        method=embedding_method,
                        gamma=embedding_gamma,
                        max_hops=embedding_max_hops,
                        force_rebuild=force_rebuild,
                        folder_path=logic_text_folder,
                        pattern=logic_text_pattern
                    )
                
                futures['embed'] = executor.submit(embed_nodes_task)
                
                for name, future in futures.items():
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  [错误] 任务 {name} 失败: {e}")
                        raise
        else:
            if chunks:
                self.extract_keywords(chunks, force_rebuild=force_rebuild)
            
            self.embed_graph_nodes(
                method=embedding_method,
                gamma=embedding_gamma,
                max_hops=embedding_max_hops,
                force_rebuild=force_rebuild,
                folder_path=logic_text_folder,
                pattern=logic_text_pattern
            )

    def _run_phase3_mounting(
        self,
        documents_folder: Optional[str],
        force_rebuild: bool,
        debug_enrichment: bool,
    ):
        from src.config import LOG_VERBOSE
        if documents_folder:
            skip_mounting = False
            only_new_nodes = False
            
            if not force_rebuild:
                skip_mounting = self.build_cache.should_skip_knowledge_mounting(
                    graph=self.graph_store.graph
                )
                
                if not skip_mounting and self.build_cache.stage2_cache.get("knowledge_mounted", False):
                    only_new_nodes = True
            
            if skip_mounting:
                if LOG_VERBOSE:
                    print("      [缓存] 所有节点都已挂载，跳过知识挂载")
            else:
                if only_new_nodes and LOG_VERBOSE:
                    print("      [增量挂载] 只挂载新节点...")
                
                self.mount_knowledge(debug=debug_enrichment, only_new_nodes=only_new_nodes)
                
                self.build_cache.update_stage2_cache(
                    knowledge_mounted=True,
                )

    def build(
        self,
        logic_text_folder: Optional[str] = None,
        documents_folder: Optional[str] = None,
        document_pattern: str = "*.txt",
        logic_text_pattern: str = "*.txt",
        skip_graph_build: bool = False,
        saved_graph_json: Optional[str] = None,
        debug_enrichment: bool = False,
        force_rebuild: bool = False,
        # 两阶段模式覆盖参数（可选，覆盖构造函数中的设置）
        use_two_stage: Optional[bool] = None,
        embedding_method: Optional[str] = None,
        embedding_gamma: float = 0.5,
        embedding_max_hops: int = 2,
        # 并发控制参数
        enable_parallel: bool = True,
        max_workers: int = 2,
    ) -> Dict[str, Any]:
        """构建知识库（三阶段并发：图谱+文档并行 → 关键词+嵌入并行 → 知识挂载）"""
        from src.config import LOG_VERBOSE
        self.processed_file_records = []
        
        if embedding_method is None:
            embedding_method = "residual"
        
        use_two_stage_mode = use_two_stage if use_two_stage is not None else self.two_stage_mode
        
        # ========== 阶段1: 图谱构建 + 文档处理 ==========
        print("\n========== 阶段1: 图谱构建和文档处理 ==========")
        
        if enable_parallel and documents_folder:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                
                def build_graph_task():
                    self._build_or_load_graph(
                        skip_graph_build=skip_graph_build,
                        saved_graph_json=saved_graph_json,
                        logic_text_folder=logic_text_folder,
                        logic_text_pattern=logic_text_pattern,
                        force_rebuild=force_rebuild,
                        max_workers=max_workers,
                        use_two_stage_mode=use_two_stage_mode,
                    )
                
                def process_docs_task():
                    return self.process_documents(documents_folder, document_pattern, force_rebuild=force_rebuild)
                
                futures['graph'] = executor.submit(build_graph_task)
                futures['docs'] = executor.submit(process_docs_task)
                
                chunks = []
                for name, future in futures.items():
                    try:
                        if name == 'docs':
                            chunks = future.result()
                        else:
                            future.result()
                    except Exception as e:
                        print(f"  [错误] 任务 {name} 失败: {e}")
                        raise
        else:
            self._build_or_load_graph(
                skip_graph_build=skip_graph_build,
                saved_graph_json=saved_graph_json,
                logic_text_folder=logic_text_folder,
                logic_text_pattern=logic_text_pattern,
                force_rebuild=force_rebuild,
                max_workers=max_workers,
                use_two_stage_mode=use_two_stage_mode,
            )
            
            chunks = []
            if documents_folder:
                chunks = self.process_documents(documents_folder, document_pattern, force_rebuild=force_rebuild)
        
        if force_rebuild:
            self.build_cache.clear_stage2_cache()
        
        # ========== 阶段2: 关键词提取 + 节点嵌入 ==========
        print("\n========== 阶段2: 关键词提取和节点嵌入 ==========")
        
        self._run_phase2(
            chunks=chunks,
            documents_folder=documents_folder,
            document_pattern=document_pattern,
            logic_text_folder=logic_text_folder,
            logic_text_pattern=logic_text_pattern,
            force_rebuild=force_rebuild,
            embedding_method=embedding_method,
            embedding_gamma=embedding_gamma,
            embedding_max_hops=embedding_max_hops,
            enable_parallel=enable_parallel,
            max_workers=max_workers,
        )
        
        # ========== 阶段3: 知识挂载 ==========
        print("\n========== 阶段3: 知识挂载 ==========")
        
        self._run_phase3_mounting(
            documents_folder=documents_folder,
            force_rebuild=force_rebuild,
            debug_enrichment=debug_enrichment,
        )
        
        print("\n========== [6/6] 构建完成! ==========")
        print(f"统计信息: {json.dumps(self.stats, ensure_ascii=False, indent=2)}")
        
        if LOG_VERBOSE:
            self.token_stats.print_summary()
        self.stats["tokens"] = self.token_stats.to_dict()
        
        if self.processed_file_records:
            for record in self.processed_file_records:
                self.build_cache.add_file_record(
                    file_path=record["file_path"],
                    tg_id=record["tg_id"],
                    nodes_count=record["nodes_count"],
                    edges_count=record["edges_count"],
                    success=record["success"],
                    error_message=record.get("error_message"),
                )
            self.processed_file_records.clear()
        
        self.build_cache.update_token_statistics(self.token_stats.to_dict())
        self.build_cache.save()
        if LOG_VERBOSE:
            print(f"      [缓存] 已保存到: {self.build_cache.cache_file}")
            self.build_cache.print_statistics()
        
        stats_path = self.persist_directory / "build_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        return self.stats
    
    def get_graph_store(self) -> GraphStore:
        return self.graph_store
    
    def get_vector_db(self) -> VectorDB:
        return self.vector_db
    
    def get_embedder(self) -> Embedder:
        return self.embedder
    
    def get_keyword_index(self) -> KeywordIndex:
        return self.keyword_index
    
    def get_knowledge_mounter(self) -> KnowledgeMounter:
        return self.knowledge_mounter
    
    def close(self):
        self.graph_store.close()
