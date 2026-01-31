"""LLM 客户端模块
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import yaml

try:
    import tiktoken
except ImportError:
    # tiktoken 未安装时使用降级方案
    tiktoken = None


# 环境变量匹配模式，用于配置文件中的 ${VAR_NAME} 替换
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# 默认定价配置（美元/1k tokens）
DEFAULT_PRICING = {
    "claude-3-5-sonnet-20241022": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-3-opus-20240229": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "claude-3-haiku-20240307": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
    "claude-3-5-haiku-20241022": {"input_per_1k": 0.001, "output_per_1k": 0.005},
    # 兼容旧配置
    "gpt-4": {"input_per_1k": 0.03, "output_per_1k": 0.06},
    "gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
}


class RetryableError(Exception):
    """可重试的错误
    
    当 API 返回 429（速率限制）或 5xx（服务器错误）时抛出此异常，
    触发指数退避重试机制。
    """


class MaxRetryExceededError(Exception):
    """超过最大重试次数
    
    当重试次数达到配置的上限后仍然失败时抛出此异常。
    """


@dataclass
class TokenStats:
    """Token 统计数据结构
    
    记录 LLM 调用的 Token 消耗和成本估算。
    
    Attributes:
        total_calls: 总调用次数
        total_input_tokens: 累计输入 Token 数
        total_output_tokens: 累计输出 Token 数
        total_tokens: 累计总 Token 数
        estimated_cost: 预估成本（美元）
        last_input_tokens: 最近一次调用的输入 Token 数
        last_output_tokens: 最近一次调用的输出 Token 数
        last_total_tokens: 最近一次调用的总 Token 数
    """

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    total_output_chars: int = 0

    # 最近一次调用的统计
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_total_tokens: int = 0
    last_output_chars: int = 0


class LLMClient:
    """LLM 客户端，支持 Claude API
    
    提供同步调用方式，支持自动重试、Token 统计等功能。
    
    Example:
        >>> client = LLMClient(config_path="configs/llm.yaml")
        >>> response = client.generate(
        ...     system_prompt="你是一个有帮助的助手",
        ...     user_prompt="你好"
        ... )
        >>> print(response)
        >>> client.print_token_stats()
    """

    def __init__(
        self,
        config_path: str = "configs/llm.yaml",
        http_client: Optional[httpx.Client] = None,
        logger: Optional[Any] = None,
    ) -> None:
        """从配置文件初始化客户端
        
        Args:
            config_path: 配置文件路径，支持相对路径和绝对路径
            http_client: 可选的同步 HTTP 客户端（用于测试注入）
        
        Raises:
            ValueError: 当 api_key 未配置时抛出
        """
        self._config = self._load_config(config_path)
        llm_cfg = self._config.get("llm", self._config)

        # 基础配置
        self.provider = llm_cfg.get("provider", "claude")
        self.model_name = llm_cfg.get("model_name", "claude-3-5-sonnet-20241022")
        self.base_url = llm_cfg.get("base_url", "https://api.anthropic.com/v1")
        self.api_key = llm_cfg.get("api_key", "")
        self.api_version = llm_cfg.get("api_version", "2023-06-01")
        self.max_tokens = int(llm_cfg.get("max_tokens", 4096))
        self.temperature = float(llm_cfg.get("temperature", 0.7))
        self.timeout = float(llm_cfg.get("timeout", 60))

        # 重试配置
        self.retry_count = int(llm_cfg.get("retry_count", 3))
        self.retry_base_delay = float(llm_cfg.get("retry_base_delay", 1.0))
        self.retry_max_delay = float(llm_cfg.get("retry_max_delay", 60))
        
        # 代理配置（可选）
        self.proxy_url = llm_cfg.get("proxy_url", None)

        if not self.api_key:
            raise ValueError("api_key 未配置，请在 llm.yaml 中设置或通过环境变量提供")

        # 定价配置（合并默认配置和用户自定义配置）
        self._pricing = dict(DEFAULT_PRICING)
        self._pricing.update(llm_cfg.get("pricing", {}) or {})

        # Token 统计
        self._stats = TokenStats()
        
        # HTTP 客户端（支持注入，便于测试）
        self._client = http_client
        self._logger = logger

    # ==================== 公开 API ====================

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """同步生成文本
        
        Args:
            system_prompt: 系统提示词，用于设定 AI 的角色和行为
            user_prompt: 用户提示词，即用户的输入
            
        Returns:
            AI 生成的文本响应
            
        Raises:
            MaxRetryExceededError: 超过最大重试次数
            httpx.HTTPStatusError: HTTP 请求失败（非重试状态码）
        """
        messages = [{"role": "user", "content": user_prompt}]
        return self._with_retry_sync(messages, system_prompt)

    def chat(self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """多轮对话
        
        Args:
            messages: 对话历史，格式为 [{"role": "user/assistant", "content": "..."}]
            system_prompt: 可选的系统提示词
            
        Returns:
            AI 生成的响应文本
        """
        return self._with_retry_sync(messages, system_prompt)

    def get_token_stats(self) -> TokenStats:
        """获取 Token 统计数据
        
        Returns:
            包含累计和最近一次调用统计的 TokenStats 对象
        """
        return self._stats

    def print_token_stats(self) -> None:
        """打印美观的 Token 统计信息到控制台
        
        输出格式化的统计表格，包括本次调用和累计统计。
        """
        s = self._stats
        lines = [
            "┌─────────────────────────────────────────────────────────┐",
            "│                    LLM Token 统计                        │",
            "├─────────────────────────────────────────────────────────┤",
            f"│  模型: {self.model_name:<50}│",
            "│  ─────────────────────────────────────────────────────  │",
            "│  本次调用:                                              │",
            f"│    Input Tokens:   {s.last_input_tokens:>8,}                            │",
            f"│    Output Tokens:  {s.last_output_tokens:>8,}                            │",
            f"│    Total Tokens:   {s.last_total_tokens:>8,}                            │",
            f"│    Output Chars:   {s.last_output_chars:>8,}                            │",
            "│  ─────────────────────────────────────────────────────  │",
            "│  累计统计:                                              │",
            f"│    总调用次数:     {s.total_calls:>8,}                            │",
            f"│    累计 Tokens:    {s.total_tokens:>8,}                            │",
            f"│    累计 Chars:     {s.total_output_chars:>8,}                            │",
            f"│    预估成本:       ${s.estimated_cost:>7.4f}                           │",
            "└─────────────────────────────────────────────────────────┘",
        ]
        print("\n".join(lines))

    def reset_stats(self) -> None:
        """重置 Token 统计
        
        清空所有累计统计数据，用于开始新的统计周期。
        """
        self._stats = TokenStats()
    
    def set_logger(self, logger: Optional[Any]) -> None:
        """设置日志记录器（可选）"""
        self._logger = logger

    def close(self) -> None:
        """关闭同步 HTTP 客户端
        
        释放网络连接资源，建议在使用完毕后调用。
        """
        if self._client is not None:
            self._client.close()

    # ==================== 配置加载 ====================

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            解析后的配置字典，环境变量已替换
        """
        path = config_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return self._expand_env_vars(data)

    def _expand_env_vars(self, value: Any) -> Any:
        """递归替换配置中的环境变量
        
        支持 ${VAR_NAME} 格式的环境变量引用。
        
        Args:
            value: 配置值（可以是字典、列表或字符串）
            
        Returns:
            替换环境变量后的值
        """
        if isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._expand_env_vars(v) for v in value]
        if isinstance(value, str):
            def _replace(match: re.Match) -> str:
                env_name = match.group(1)
                return os.getenv(env_name, "")

            return _ENV_VAR_PATTERN.sub(_replace, value)
        return value

    # ==================== HTTP 请求 ====================

    def _get_headers(self) -> Dict[str, str]:
        """构建 HTTP 请求头
        
        根据 provider 类型返回对应的认证头。
        """
        if self.provider == "claude":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": "application/json",
            }
        # OpenAI 兼容格式
        return {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def _get_endpoint(self) -> str:
        """获取 API 端点路径"""
        if self.provider == "claude":
            return "/messages"
        return "/chat/completions"

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
    ) -> Dict[str, Any]:
        """构建 API 请求体
        
        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            
        Returns:
            API 请求的 JSON 负载
        """
        if self.provider == "claude":
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages,
            }
            if system_prompt:
                payload["system"] = system_prompt
            return payload

        # OpenAI 兼容格式
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            payload["messages"] = [{"role": "system", "content": system_prompt}] + messages
        return payload

    def _get_client(self) -> httpx.Client:
        """获取或创建同步 HTTP 客户端"""
        if self._client is None:
            # 处理代理配置
            if self.proxy_url:
                # 将 socks5h:// 转换为 socks5://（httpx 只支持 socks5://）
                proxy = self.proxy_url.replace("socks5h://", "socks5://")
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    proxy=proxy,
                )
            else:
                # 不使用代理，禁用环境变量代理（避免 socks5h:// 不兼容问题）
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    trust_env=False,  # 禁用环境变量代理
                )
        return self._client

    # ==================== 重试逻辑 ====================

    def _with_retry_sync(self, messages: List[Dict[str, Any]], system_prompt: Optional[str]) -> str:
        """带重试的同步 API 调用
        
        使用指数退避策略重试失败的请求。
        延迟计算公式: delay = base_delay * (2 ^ attempt) + random(0, 1)
        
        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            
        Returns:
            API 响应文本
            
        Raises:
            MaxRetryExceededError: 超过最大重试次数
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_count):
            try:
                return self._call_api_sync(messages, system_prompt)
            except RetryableError as exc:
                last_exc = exc
                delay = min(
                    self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.retry_max_delay,
                )
                time.sleep(delay)
        raise MaxRetryExceededError(str(last_exc) if last_exc else "未知错误")

    def _call_api_sync(self, messages: List[Dict[str, Any]], system_prompt: Optional[str]) -> str:
        """同步调用 API
        
        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            
        Returns:
            提取的响应文本
            
        Raises:
            RetryableError: 遇到可重试的 HTTP 错误（429, 5xx）
            httpx.HTTPStatusError: 其他 HTTP 错误
        """
        payload = self._build_payload(messages, system_prompt)
        response = self._get_client().post(
            self._get_endpoint(),
            json=payload,
            headers=self._get_headers(),
        )
        # 可重试的状态码
        if response.status_code in {429, 500, 502, 503, 504}:
            raise RetryableError(f"HTTP {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        self._update_stats(data.get("usage"), system_prompt, messages, text)
        return text

    # ==================== 响应处理 ====================

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """从 API 响应中提取文本内容
        
        支持 Claude 和 OpenAI 两种响应格式。
        
        Args:
            data: API 返回的 JSON 数据
            
        Returns:
            提取的文本内容
        """
        if self.provider == "claude":
            content = data.get("content", [])
            if isinstance(content, list):
                parts = [
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                return "".join(parts)
            if isinstance(content, str):
                return content
        # OpenAI 兼容格式
        choices = data.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "")
        return ""

    # ==================== Token 统计 ====================

    def _update_stats(
        self,
        usage: Optional[Dict[str, Any]],
        system_prompt: Optional[str],
        messages: List[Dict[str, Any]],
        output_text: str,
    ) -> None:
        """更新 Token 统计
        
        优先使用 API 返回的 usage 数据，如果不可用则使用 tiktoken 估算。
        
        Args:
            usage: API 返回的 usage 字段
            system_prompt: 系统提示词
            messages: 对话消息列表
            output_text: 输出文本
        """
        input_tokens = None
        output_tokens = None
        
        # 优先使用 API 返回的 token 数量
        if isinstance(usage, dict):
            # Claude 格式: input_tokens, output_tokens
            # OpenAI 格式: prompt_tokens, completion_tokens
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")

        # 降级使用 tiktoken 估算
        if input_tokens is None:
            input_tokens = self._estimate_input_tokens(system_prompt, messages)
        if output_tokens is None:
            output_tokens = self._estimate_tokens(output_text)

        total_tokens = input_tokens + output_tokens
        output_chars = len(output_text or "")
        
        # 更新统计数据
        self._stats.total_calls += 1
        self._stats.total_input_tokens += input_tokens
        self._stats.total_output_tokens += output_tokens
        self._stats.total_tokens = self._stats.total_input_tokens + self._stats.total_output_tokens
        self._stats.last_input_tokens = input_tokens
        self._stats.last_output_tokens = output_tokens
        self._stats.last_total_tokens = total_tokens
        self._stats.total_output_chars += output_chars
        self._stats.last_output_chars = output_chars
        self._stats.estimated_cost += self._estimate_cost(input_tokens, output_tokens)
        
        if self._logger:
            self._logger.info(
                "LLM output chars: %s (input_tokens=%s, output_tokens=%s)",
                output_chars,
                input_tokens,
                output_tokens,
            )

    def _estimate_input_tokens(self, system_prompt: Optional[str], messages: List[Dict[str, Any]]) -> int:
        """估算输入文本的 Token 数量
        
        将系统提示词和所有消息内容合并后进行估算。
        
        Args:
            system_prompt: 系统提示词
            messages: 对话消息列表
            
        Returns:
            估算的 Token 数量
        """
        parts: List[str] = []
        if system_prompt:
            parts.append(system_prompt)
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
            elif isinstance(content, str):
                parts.append(content)
        return self._estimate_tokens("\n".join(parts))

    def _estimate_tokens(self, text: str) -> int:
        """使用 tiktoken 估算文本的 Token 数量
        
        Claude 模型使用与 GPT-4 相似的分词器，tiktoken 的 cl100k_base 编码
        可以提供较为准确的估算。如果 tiktoken 未安装，则降级为简单的空格分词。
        
        Args:
            text: 待估算的文本
            
        Returns:
            估算的 Token 数量
        """
        if not text:
            return 0
        if tiktoken is None:
            # 降级方案：简单按空格分词
            return max(1, len(text.split()))
        try:
            # Claude 使用 cl100k_base 编码
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # 出错时使用降级方案
            return max(1, len(text.split()))

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """估算 API 调用成本
        
        根据模型定价计算本次调用的成本（美元）。
        
        Args:
            input_tokens: 输入 Token 数量
            output_tokens: 输出 Token 数量
            
        Returns:
            估算成本（美元），如果模型未配置定价则返回 0.0
        """
        price = self._pricing.get(self.model_name)
        if not price:
            return 0.0
        in_rate = float(price.get("input_per_1k", 0.0))
        out_rate = float(price.get("output_per_1k", 0.0))
        return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
