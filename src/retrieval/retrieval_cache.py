# -*- coding: utf-8 -*-
"""
检索缓存管理器

记录已检索的问题和结果，支持断点续传和重复检索跳过。

Author: CongCongTian
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class RetrievalRecord:
    """检索记录"""
    
    def __init__(
        self,
        question: str,
        question_hash: str,
        answer: str,
        retrieved_content: List[Dict[str, Any]],
        merged_context: str,
        reasoning_path: List[Any],
        token_usage: int,
        retrieval_time: str,
        status: str = "success",
        error: Optional[str] = None,
        llm_prompt: Optional[Dict[str, Any]] = None,
    ):
        self.question = question
        self.question_hash = question_hash
        self.answer = answer
        self.retrieved_content = retrieved_content
        self.merged_context = merged_context
        self.reasoning_path = reasoning_path
        self.token_usage = token_usage
        self.retrieval_time = retrieval_time
        self.status = status
        self.error = error
        # 完整的 LLM 输入 prompt，格式:
        # {
        #   "system_prompt": "...",      完整 system prompt（含 context_data）
        #   "user_prompt":   "...",      用户问题
        #   "system_prompt_tokens": 123, cl100k_base token 数
        #   "user_prompt_tokens":   45,
        # }
        self.llm_prompt: Dict[str, Any] = llm_prompt or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question": self.question,
            "question_hash": self.question_hash,
            "answer": self.answer,
            "retrieved_content": self.retrieved_content,
            "merged_context": self.merged_context,
            "reasoning_path": self.reasoning_path,
            "token_usage": self.token_usage,
            "retrieval_time": self.retrieval_time,
            "status": self.status,
            "error": self.error,
            "llm_prompt": self.llm_prompt,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalRecord":
        """从字典创建"""
        return cls(
            question=data["question"],
            question_hash=data["question_hash"],
            answer=data["answer"],
            retrieved_content=data.get("retrieved_content", []),
            merged_context=data.get("merged_context", ""),
            reasoning_path=data.get("reasoning_path", []),
            token_usage=data.get("token_usage", 0),
            retrieval_time=data["retrieval_time"],
            status=data.get("status", "success"),
            error=data.get("error"),
            llm_prompt=data.get("llm_prompt", {}),
        )


class RetrievalCache:
    """检索缓存管理器
    
    基于问题哈希去重，支持断点续传和失败重试。
    """
    
    VERSION = "1.0"
    
    def __init__(self, cache_file: str):
        """
        Args:
            cache_file: 缓存文件路径
        """
        self.cache_file = Path(cache_file)
        self.questions: Dict[str, RetrievalRecord] = {}
        self.statistics: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        """从文件加载缓存"""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if data.get("version") != self.VERSION:
                from src.config import LOG_VERBOSE
                if LOG_VERBOSE:
                    print(f"      [检索缓存] 版本不匹配，重置缓存")
                return
            
            self.questions = {
                question_hash: RetrievalRecord.from_dict(record)
                for question_hash, record in data.get("questions", {}).items()
            }
            
            self.statistics = data.get("statistics", {})
            
            from src.config import LOG_VERBOSE
            if LOG_VERBOSE:
                print(f"      [检索缓存] 已加载 {len(self.questions)} 条检索记录")
        
        except Exception as e:
            print(f"      [检索缓存] 加载失败: {e}，将创建新缓存")
            self.questions = {}
            self.statistics = {}
    
    def save(self):
        """持久化缓存到文件"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._update_statistics()
        
        data = {
            "version": self.VERSION,
            "last_update": datetime.now().isoformat(),
            "statistics": self.statistics,
            "questions": {
                question_hash: record.to_dict()
                for question_hash, record in self.questions.items()
            }
        }
        
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _update_statistics(self):
        """更新统计信息"""
        records = list(self.questions.values())
        successful = [r for r in records if r.status == "success"]
        
        self.statistics = {
            "total_questions": len(records),
            "successful_questions": len(successful),
            "failed_questions": len(records) - len(successful),
            "total_tokens": sum(r.token_usage for r in records),
            "avg_tokens_per_question": (
                sum(r.token_usage for r in successful) / len(successful)
                if successful else 0
            ),
        }
    
    _FAKE_SUCCESS_ANSWERS = frozenset({
        "无法回答", "无法生成答案", "",
        "根据当前可用数据，无法找到与该问题直接相关的信息。",
    })
    
    def _is_fake_success(self, record: 'RetrievalRecord') -> bool:
        """判断 success 记录是否为假成功（LLM 空响应或默认无数据回复）"""
        if record.status != "success":
            return False
        ans = (record.answer or "").strip()
        if ans in self._FAKE_SUCCESS_ANSWERS:
            return True
        if (ans.startswith("生成答案时出错:") 
                or ans.startswith("无法生成答案：")
                or ans.startswith("假成功-")
                or ans.startswith("无法回答：")):
            return True
        if "I am sorry but I am unable to answer" in ans:
            return True
        return False
    
    def _needs_retry(self, record: 'RetrievalRecord') -> bool:
        """判断一条记录是否需要重跑"""
        if record.status == "error":
            return True
        if self._is_fake_success(record):
            return True
        return False
    
    def is_question_cached(self, question: str, treat_error_as_cached: bool = False) -> bool:
        """检查问题是否已成功缓存（error/假成功默认视为未缓存）"""
        question_hash = self._compute_question_hash(question)
        if question_hash not in self.questions:
            return False
        record = self.questions[question_hash]
        if not treat_error_as_cached and self._needs_retry(record):
            return False
        return True
    
    def get_cached_result(self, question: str) -> Optional[RetrievalRecord]:
        """获取已缓存的检索结果，不存在返回 None"""
        question_hash = self._compute_question_hash(question)
        return self.questions.get(question_hash)
    
    def add_retrieval_record(
        self,
        question: str,
        answer: str = "",
        retrieved_content: Optional[List[Dict[str, Any]]] = None,
        merged_context: str = "",
        reasoning_path: Optional[List[Any]] = None,
        token_usage: int = 0,
        status: str = "success",
        error: Optional[str] = None,
        llm_prompt: Optional[Dict[str, Any]] = None,
    ):
        """添加一条检索记录"""
        question_hash = self._compute_question_hash(question)
        record = RetrievalRecord(
            question=question,
            question_hash=question_hash,
            answer=answer,
            retrieved_content=retrieved_content or [],
            merged_context=merged_context,
            reasoning_path=reasoning_path or [],
            token_usage=token_usage,
            retrieval_time=datetime.now().isoformat(),
            status=status,
            error=error,
            llm_prompt=llm_prompt or {},
        )
        
        self.questions[question_hash] = record
    
    def clear(self):
        """清除所有缓存"""
        self.questions = {}
        self.statistics = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_cached_questions(self) -> List[str]:
        """获取所有已缓存的问题"""
        return [record.question for record in self.questions.values()]
    
    def get_questions_to_process(self, all_questions: List[str], retry_errors: bool = True) -> List[str]:
        """获取需要处理的问题列表，排除已缓存的，可选重试失败条目"""
        questions_to_process = []
        
        for question in all_questions:
            question_hash = self._compute_question_hash(question)
            if question_hash not in self.questions:
                questions_to_process.append(question)
            elif retry_errors and self._needs_retry(self.questions[question_hash]):
                questions_to_process.append(question)
        
        return questions_to_process
    
    def get_failed_questions(self, error_pattern: str = None) -> List[str]:
        """获取需要重试的失败问题，可按错误关键词过滤"""
        failed = []
        for record in self.questions.values():
            if self._needs_retry(record):
                if error_pattern is None or (record.error and error_pattern in record.error):
                    failed.append(record.question)
        return failed
    
    def get_failed_statistics(self) -> Dict[str, int]:
        """按错误类型分组统计失败问题"""
        stats: Dict[str, int] = {}
        for record in self.questions.values():
            if self._needs_retry(record):
                error_msg = record.error or "unknown"
                if "429" in error_msg or "rate_limit" in error_msg or "limit_requests" in error_msg:
                    key = "rate_limit (429)"
                elif "timeout" in error_msg.lower():
                    key = "timeout"
                elif "假成功" in error_msg:
                    key = "fake_success"
                else:
                    key = "other"
                stats[key] = stats.get(key, 0) + 1
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        if not self.statistics:
            self._update_statistics()
        
        print(f"\n      [检索缓存统计]")
        print(f"        总问题数: {self.statistics.get('total_questions', 0)}")
        print(f"        成功: {self.statistics.get('successful_questions', 0)}")
        print(f"        失败: {self.statistics.get('failed_questions', 0)}")
        print(f"        总Token消耗: {self.statistics.get('total_tokens', 0)}")
        print(f"        平均每问题: {self.statistics.get('avg_tokens_per_question', 0):.2f} tokens")
    
    @staticmethod
    def _compute_question_hash(question: str) -> str:
        """计算问题的 MD5 哈希值"""
        normalized = question.strip().replace('\r\n', '\n')
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
