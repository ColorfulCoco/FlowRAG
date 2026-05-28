# -*- coding: utf-8 -*-
"""
OpenAI 客户端工厂与 API 限流器

提供连接池管理、超时控制、自动重试、429 限流退避等高并发场景下的基础设施。

Author: CongCongTian
"""

import time
import random
import threading
import httpx
from openai import OpenAI
from typing import Optional


DEFAULT_TIMEOUT = None
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CONNECTIONS = 200
DEFAULT_MAX_KEEPALIVE = 40


def create_openai_client(
    api_key: str,
    base_url: str,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
) -> OpenAI:
    """
    创建带连接池和重试策略的 OpenAI 客户端

    Args:
        api_key: API 密钥
        base_url: API 端点
        timeout: 读取超时（秒），None 表示不限制
        connect_timeout: 连接超时（秒）
        max_retries: HTTP 层自动重试次数
        max_connections: 连接池最大连接数
        max_keepalive: 长连接保持数
    """
    http_client = httpx.Client(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        ),
        timeout=httpx.Timeout(
            timeout=timeout,
            connect=connect_timeout,
        ),
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client,
    )
    return client


def check_finish_reason(response, caller: str = "") -> Optional[str]:
    """检查 API 响应的 finish_reason，finish_reason=length 时打印截断警告"""
    try:
        if not response or not hasattr(response, 'choices') or not response.choices:
            return None
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            tag = f"[{caller}] " if caller else ""
            print(f"      {tag}[警告] 响应被截断 (finish_reason=length)，输出可能不完整")
        return finish_reason
    except Exception:
        return None


class APIRateLimiter:
    """
    API 限流器：信号量控制并发 + 429 全局冷却 + 退避抖动

    用法:
        limiter = APIRateLimiter(max_concurrent=16)
        with limiter.acquire():
            response = client.chat.completions.create(...)
    """

    def __init__(self, max_concurrent: int = 16, base_cooldown: float = 60.0):
        """
        Args:
            max_concurrent: 最大并发请求数
            base_cooldown: 429 后基础冷却秒数，实际等待 = base * (1 +/- 0.3 抖动)
        """
        self._semaphore = threading.Semaphore(max_concurrent)
        self._base_cooldown = base_cooldown
        self._cooldown_until = 0.0
        self._cooldown_lock = threading.Lock()
        self._consecutive_429 = 0
        self._max_concurrent = max_concurrent

    class _AcquireContext:
        def __init__(self, limiter: 'APIRateLimiter'):
            self._limiter = limiter

        def __enter__(self):
            self._limiter._wait_if_cooling()
            self._limiter._semaphore.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None and _is_rate_limit_error(exc_val):
                self._limiter.report_rate_limit()
            self._limiter._semaphore.release()
            return False

    def acquire(self) -> '_AcquireContext':
        """获取限流许可，配合 with 语句使用"""
        return self._AcquireContext(self)

    def report_rate_limit(self):
        """收到 429 后触发全局冷却"""
        with self._cooldown_lock:
            self._consecutive_429 += 1
            multiplier = min(self._consecutive_429, 6)
            cooldown = self._base_cooldown * multiplier
            jitter = cooldown * 0.3 * (2 * random.random() - 1)  # ±30%
            actual_wait = cooldown + jitter
            new_deadline = time.time() + actual_wait
            if new_deadline > self._cooldown_until:
                self._cooldown_until = new_deadline
                print(f"      [限流器] 收到 429 (连续第{self._consecutive_429}次)，"
                      f"全局冷却 {actual_wait:.1f}s (并发上限: {self._max_concurrent})")

    def reset_429_counter(self):
        """请求成功时调用，重置连续 429 计数"""
        with self._cooldown_lock:
            if self._consecutive_429 > 0:
                self._consecutive_429 = 0

    def _wait_if_cooling(self):
        """冷却期内阻塞等待，加随机抖动让各线程错开恢复"""
        while True:
            now = time.time()
            with self._cooldown_lock:
                remaining = self._cooldown_until - now
            if remaining <= 0:
                break
            jitter = random.uniform(0, min(remaining * 0.2, 2.0))
            time.sleep(remaining + jitter)


def _is_rate_limit_error(exc) -> bool:
    """判断异常是否为 429 限流"""
    if exc is None:
        return False
    exc_str = str(exc)
    if "429" in exc_str:
        return True
    if "rate_limit" in exc_str.lower() or "insufficient_quota" in exc_str.lower():
        return True
    err_type = type(exc).__name__
    if "RateLimitError" in err_type:
        return True
    return False


def is_rate_limit_error(exc) -> bool:
    """判断异常是否为 429 限流（公开接口）"""
    return _is_rate_limit_error(exc)


def call_with_retry(fn, *, interval_seconds: Optional[float] = None, max_attempts: Optional[int] = None, caller: str = ""):
    """
    带重试的 API 调用包装器，异常时等待后重试

    Args:
        fn: 无参可调用对象
        interval_seconds: 重试间隔秒数，None 则从 config 读取
        max_attempts: 最大重试次数（不含首次），None 则从 config 读取
        caller: 调用者标识，用于日志
    """
    try:
        from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS
    except ImportError:
        API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS = 10, 10
    _interval = interval_seconds if interval_seconds is not None else API_RETRY_INTERVAL_SECONDS
    _max = max_attempts if max_attempts is not None else API_RETRY_MAX_ATTEMPTS
    last_error = None
    for attempt in range(_max + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < _max:
                tag = f"[{caller}] " if caller else ""
                print(f"      {tag}重试 {attempt + 1}/{_max + 1}: {e} ({_interval}s后重试)")
                time.sleep(_interval)
            else:
                raise
