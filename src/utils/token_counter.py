# -*- coding: utf-8 -*-
"""
Token 用量统计

基于 API response.usage 字段统计 LLM 和 Embedding 的 token 消耗，与实际计费一致。

Author: CongCongTian
"""


def format_token_count(count: int) -> str:
    """将 token 数量格式化为易读字符串（如 1.23K、1.23M）"""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.2f}K"
    else:
        return f"{count / 1_000_000:.2f}M"


def estimate_cost(token_count: int, cost_per_million: float) -> float:
    """按 token_count / 1M * cost_per_million 估算费用（元）"""
    return (token_count / 1_000_000) * cost_per_million


class TokenStatistics:
    """按类型分类的 token 消耗统计，区分本次构建与历史累计"""
    
    def __init__(self):
        self.graph_generation_tokens = 0
        self.graph_generation_prompt_tokens = 0
        self.graph_generation_completion_tokens = 0
        
        self.text_embedding_tokens = 0
        self.keyword_embedding_tokens = 0
        self.node_embedding_tokens = 0
        
        self.cumulative_graph_generation_tokens = 0
        self.cumulative_graph_generation_prompt_tokens = 0
        self.cumulative_graph_generation_completion_tokens = 0
        
        self.cumulative_text_embedding_tokens = 0
        self.cumulative_keyword_embedding_tokens = 0
        self.cumulative_node_embedding_tokens = 0
    
    def add_graph_generation(self, tokens: int, prompt_tokens: int = 0, completion_tokens: int = 0):
        """累加图谱生成（LLM）的 token 消耗"""
        self.graph_generation_tokens += tokens
        self.cumulative_graph_generation_tokens += tokens
        
        if prompt_tokens > 0:
            self.graph_generation_prompt_tokens += prompt_tokens
            self.cumulative_graph_generation_prompt_tokens += prompt_tokens
        
        if completion_tokens > 0:
            self.graph_generation_completion_tokens += completion_tokens
            self.cumulative_graph_generation_completion_tokens += completion_tokens
    
    def add_text_embedding(self, tokens: int):
        """累加文本 Embedding 的 token 消耗"""
        self.text_embedding_tokens += tokens
        self.cumulative_text_embedding_tokens += tokens
    
    def add_keyword_embedding(self, tokens: int):
        """累加关键词 Embedding 的 token 消耗"""
        self.keyword_embedding_tokens += tokens
        self.cumulative_keyword_embedding_tokens += tokens
    
    def add_node_embedding(self, tokens: int):
        """累加图谱节点 Embedding 的 token 消耗"""
        self.node_embedding_tokens += tokens
        self.cumulative_node_embedding_tokens += tokens
    
    def get_current_total(self) -> int:
        """获取本次构建的总 token 消耗"""
        return (
            self.graph_generation_tokens +
            self.text_embedding_tokens +
            self.keyword_embedding_tokens +
            self.node_embedding_tokens
        )
    
    def get_cumulative_total(self) -> int:
        """获取累计总 token 消耗"""
        return (
            self.cumulative_graph_generation_tokens +
            self.cumulative_text_embedding_tokens +
            self.cumulative_keyword_embedding_tokens +
            self.cumulative_node_embedding_tokens
        )
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "current": {
                "graph_generation": self.graph_generation_tokens,
                "graph_generation_prompt": self.graph_generation_prompt_tokens,
                "graph_generation_completion": self.graph_generation_completion_tokens,
                "text_embedding": self.text_embedding_tokens,
                "keyword_embedding": self.keyword_embedding_tokens,
                "node_embedding": self.node_embedding_tokens,
                "total": self.get_current_total(),
            },
            "cumulative": {
                "graph_generation": self.cumulative_graph_generation_tokens,
                "graph_generation_prompt": self.cumulative_graph_generation_prompt_tokens,
                "graph_generation_completion": self.cumulative_graph_generation_completion_tokens,
                "text_embedding": self.cumulative_text_embedding_tokens,
                "keyword_embedding": self.cumulative_keyword_embedding_tokens,
                "node_embedding": self.cumulative_node_embedding_tokens,
                "total": self.get_cumulative_total(),
            },
        }
    
    def print_summary(self):
        """打印本次构建和历史累计的 token 用量及成本估算"""
        _fmt = format_token_count
        print("\n" + "=" * 70)
        print("Token 消耗统计（来源: API response.usage）")
        print("=" * 70)
        
        print("\n【本次构建】")
        print(f"  图谱生成 (LLM):        {_fmt(self.graph_generation_tokens):>10} tokens")
        if self.graph_generation_prompt_tokens > 0 or self.graph_generation_completion_tokens > 0:
            print(f"    ├─ 输入 (Prompt):    {_fmt(self.graph_generation_prompt_tokens):>10} tokens")
            print(f"    └─ 输出 (Completion):{_fmt(self.graph_generation_completion_tokens):>10} tokens")
        print(f"  文本 Embedding:        {_fmt(self.text_embedding_tokens):>10} tokens")
        print(f"  关键词 Embedding:      {_fmt(self.keyword_embedding_tokens):>10} tokens")
        print(f"  图谱节点 Embedding:    {_fmt(self.node_embedding_tokens):>10} tokens")
        print(f"  ────────────────────────────────────")
        print(f"  本次总计:              {_fmt(self.get_current_total()):>10} tokens")
        
        print("\n【累计统计（含历史）】")
        print(f"  图谱生成 (LLM):        {_fmt(self.cumulative_graph_generation_tokens):>10} tokens")
        if self.cumulative_graph_generation_prompt_tokens > 0 or self.cumulative_graph_generation_completion_tokens > 0:
            print(f"    ├─ 输入 (Prompt):    {_fmt(self.cumulative_graph_generation_prompt_tokens):>10} tokens")
            print(f"    └─ 输出 (Completion):{_fmt(self.cumulative_graph_generation_completion_tokens):>10} tokens")
        print(f"  文本 Embedding:        {_fmt(self.cumulative_text_embedding_tokens):>10} tokens")
        print(f"  关键词 Embedding:      {_fmt(self.cumulative_keyword_embedding_tokens):>10} tokens")
        print(f"  图谱节点 Embedding:    {_fmt(self.cumulative_node_embedding_tokens):>10} tokens")
        print(f"  ────────────────────────────────────")
        print(f"  累计总计:              {_fmt(self.get_cumulative_total()):>10} tokens")
        
        print("\n【成本估算】")
        print("  参考价格:")
        print(f"    LLM 输入 (Prompt):     ¥0.80 / 1M tokens")
        print(f"    LLM 输出 (Completion): ¥2.00 / 1M tokens")
        print(f"    Embedding:             ¥0.50 / 1M tokens")
        
        llm_prompt_cost = estimate_cost(self.graph_generation_prompt_tokens, 0.8)
        llm_completion_cost = estimate_cost(self.graph_generation_completion_tokens, 2.0)
        llm_cost = llm_prompt_cost + llm_completion_cost
        emb_cost = estimate_cost(
            self.text_embedding_tokens + self.keyword_embedding_tokens + self.node_embedding_tokens,
            0.5
        )
        current_total = llm_cost + emb_cost
        
        cum_llm_prompt_cost = estimate_cost(self.cumulative_graph_generation_prompt_tokens, 0.8)
        cum_llm_completion_cost = estimate_cost(self.cumulative_graph_generation_completion_tokens, 2.0)
        cum_llm_cost = cum_llm_prompt_cost + cum_llm_completion_cost
        cum_emb_cost = estimate_cost(
            self.cumulative_text_embedding_tokens +
            self.cumulative_keyword_embedding_tokens + 
            self.cumulative_node_embedding_tokens,
            0.5
        )
        cum_total = cum_llm_cost + cum_emb_cost
        
        print(f"\n  本次预估成本:          ¥{current_total:.4f}")
        print(f"    ├─ LLM 输入:         ¥{llm_prompt_cost:.4f}")
        print(f"    ├─ LLM 输出:         ¥{llm_completion_cost:.4f}")
        print(f"    └─ Embedding:        ¥{emb_cost:.4f}")
        print(f"  累计预估成本:          ¥{cum_total:.4f}")
        
        print("=" * 70 + "\n")
