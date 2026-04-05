# -*- coding: utf-8 -*-
"""
本地关键词提取器（KeyBERT + MMR）

基于 Jieba 分词 + 词性过滤生成候选词，再通过 MMR 算法在相关性
与多样性之间取得平衡，无需硬编码停用词表。

Author: CongCongTian
"""

from typing import List, Dict, Any, Iterable
from dataclasses import dataclass, field
import numpy as np

try:
    import jieba.analyse
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from src.config import LOG_VERBOSE


@dataclass
class KeywordExtractionResult:
    """关键词提取结果"""
    keywords: List[str]
    keyword_weights: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.8
    reasoning: str = "KeyBERT+MMR"


class LocalKeywordExtractor:
    """本地关键词提取器（纯语义版）
    
    通过 MMR 算法在相关性和多样性之间取平衡，替代基于规则的停用词过滤。
    """
    
    def __init__(
        self, 
        top_k: int = 5, 
        embedder = None,
        use_mmr: bool = True,
        mmr_diversity: float = 0.3,
        allowed_pos: Iterable[str] = ('n', 'nr', 'ns', 'nt', 'nz', 'eng', 'nw'),
        min_word_length: int = 2,
        candidate_source: str = "segmentation",
    ):
        """
        Args:
            embedder: Embedder 实例
            use_mmr: 是否启用 MMR 算法
            mmr_diversity: 多样性权重，0.2-0.4 为推荐范围
            candidate_source: "textrank"(TextRank 粗筛) / "segmentation"(分词+词性过滤)
        """
        if not JIEBA_AVAILABLE:
            raise ImportError("请安装jieba: pip install jieba")
        
        self.top_k = top_k
        self.embedder = embedder
        self.use_mmr = use_mmr
        self.mmr_diversity = mmr_diversity
        self.allowed_pos = tuple(allowed_pos)
        self.min_word_length = min_word_length
        self.candidate_source = candidate_source
        
        self.stats = {
            "total_calls": 0, 
            "total_keywords": 0, 
            "merged_count": 0,
        }
        
        self._embedding_cache: Dict[str, Any] = {}
    
    def _segment_candidates(self, text: str) -> List[str]:
        """通过 jieba 分词 + 词性过滤生成候选词"""
        seen = set()
        candidates = []
        
        for word, pos in pseg.cut(text):
            if not any(pos.startswith(p) for p in self.allowed_pos):
                continue
            if len(word) < self.min_word_length:
                continue
            if word in seen:
                continue
            seen.add(word)
            candidates.append(word)
        
        return candidates
    
    def _get_embedding(self, text: str):
        """获取文本嵌入向量（带内存缓存）"""
        if not self.embedder:
            return None

        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # 并发场景下需确保 embedding_type 正确
        self.embedder.set_embedding_type("keyword")
            
        vector = None
        if hasattr(self.embedder, 'embed_query'):
            vector = self.embedder.embed_query(text)
        elif hasattr(self.embedder, 'embed'):
            res = self.embedder.embed([text])
            if res:
                vector = res[0]
        
        if vector is not None:
            self._embedding_cache[text] = vector
            
        return vector

    def extract(self, text: str, debug: bool = False) -> KeywordExtractionResult:
        self.stats["total_calls"] += 1
        
        if not text or not text.strip():
            return KeywordExtractionResult(keywords=[])
        
        if self.candidate_source == "textrank":
            candidates_with_weight = jieba.analyse.textrank(
                text, 
                topK=self.top_k * 4, 
                withWeight=True, 
                allowPOS=self.allowed_pos
            )
            candidates = [w for w, _ in candidates_with_weight]
        else:
            candidates = self._segment_candidates(text)
        
        if not candidates:
            return KeywordExtractionResult(keywords=[])
            
        # 无 Embedder 时回退到 TextRank
        if not self.embedder:
            tr_keywords = jieba.analyse.textrank(text, topK=self.top_k, withWeight=True, allowPOS=self.allowed_pos)
            return KeywordExtractionResult(
                keywords=[w for w, _ in tr_keywords],
                keyword_weights={w: s for w, s in tr_keywords},
                reasoning="Fallback TextRank (No Embedder)"
            )

        try:
            doc_embedding = self._get_embedding(text[:1000])
            candidate_embeddings = []
            valid_candidates = []
            
            for w in candidates:
                emb = self._get_embedding(w)
                if emb is not None:
                    candidate_embeddings.append(emb)
                    valid_candidates.append(w)
            
            if not valid_candidates:
                return KeywordExtractionResult(keywords=[])

            # MMR 排序或纯余弦相似度
            if self.use_mmr:
                keywords = self._mmr(doc_embedding, candidate_embeddings, valid_candidates, top_n=self.top_k, diversity=self.mmr_diversity)
                reasoning = f"KeyBERT + MMR (div={self.mmr_diversity})"
            else:
                keywords = self._max_sum_similarity(doc_embedding, candidate_embeddings, valid_candidates, top_n=self.top_k)
                reasoning = "KeyBERT (Cosine Similarity)"
            
            # MMR 按顺序选出的词越靠前越好，用递减权重近似
            final_weights = {kw: 1.0 - (i * 0.1) for i, kw in enumerate(keywords)}
            
            self.stats["total_keywords"] += len(keywords)
            
            return KeywordExtractionResult(
                keywords=keywords,
                keyword_weights=final_weights,
                reasoning=reasoning
            )
            
        except Exception as e:
            print(f"[LocalKeywordExtractor] Error: {e}")
            return KeywordExtractionResult(keywords=[])

    def _prepare_similarity(self, doc_embedding, candidate_embeddings):
        """归一化文档和候选词嵌入，返回 (归一化候选矩阵, 文档-候选相似度向量)"""
        doc_emb = np.array([doc_embedding])
        cand_embs = np.array(candidate_embeddings)
        
        norm_doc = np.linalg.norm(doc_emb, axis=1, keepdims=True)
        doc_emb = doc_emb / (norm_doc + 1e-9)
        
        norm_cands = np.linalg.norm(cand_embs, axis=1, keepdims=True)
        cand_embs = cand_embs / (norm_cands + 1e-9)
        
        doc_word_sims = np.dot(cand_embs, doc_emb.T).flatten()
        return cand_embs, doc_word_sims
    
    def _mmr(self, doc_embedding, candidate_embeddings, candidates, top_n, diversity):
        """MMR 算法：在相关性和多样性之间取平衡"""
        cand_embs, doc_word_sims = self._prepare_similarity(doc_embedding, candidate_embeddings)
        
        selected_indices = []
        candidate_indices = list(range(len(candidates)))
        
        for _ in range(min(top_n, len(candidates))):
            mmr_score = -np.inf
            best_idx = -1
            
            for idx in candidate_indices:
                relevance = doc_word_sims[idx]
                
                if selected_indices:
                    current_emb = cand_embs[idx].reshape(1, -1)
                    selected_embs = cand_embs[selected_indices]
                    sims_to_selected = np.dot(current_emb, selected_embs.T).flatten()
                    max_sim_to_selected = np.max(sims_to_selected)
                else:
                    max_sim_to_selected = 0.0
                
                # MMR: (1-diversity)*Relevance - diversity*Redundancy
                score = (1 - diversity) * relevance - diversity * max_sim_to_selected
                
                if score > mmr_score:
                    mmr_score = score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
        
        return [candidates[i] for i in selected_indices]

    def _max_sum_similarity(self, doc_embedding, candidate_embeddings, candidates, top_n):
        """纯余弦相似度排序（备用）"""
        _, sims = self._prepare_similarity(doc_embedding, candidate_embeddings)
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [candidates[i] for i in top_indices]

    def extract_batch(self, texts: List[str], show_progress: bool = True, debug: bool = False) -> List[KeywordExtractionResult]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not texts:
            return []
        
        results = [None] * len(texts)  # 预分配以保持顺序
        
        def process_text(idx_text):
            idx, text = idx_text
            return idx, self.extract(text, debug=debug)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_text, (i, text)): i for i, text in enumerate(texts)}
            
            if show_progress:
                try:
                    from tqdm import tqdm
                    futures_iter = tqdm(as_completed(futures), total=len(texts), desc="关键词提取(MMR)", unit="chunk")
                except ImportError:
                    futures_iter = as_completed(futures)
            else:
                futures_iter = as_completed(futures)
            
            for future in futures_iter:
                idx, result = future.result()
                results[idx] = result
        
        return results
        
    def get_statistics(self) -> Dict:
        return {
            **self.stats,
            "avg_keywords_per_chunk": (
                self.stats["total_keywords"] / self.stats["total_calls"]
                if self.stats["total_calls"] > 0 else 0
            ),
            "method": "KeyBERT + MMR (Pure Semantic)",
            "token_cost": 0,
        }
