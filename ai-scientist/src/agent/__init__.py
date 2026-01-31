"""
Scientific Agent 模块

提供论文复现的核心组件，基于 LLM 实现代码生成与迭代优化。

主要组件:
    - Session: 会话管理，跟踪复现进度
    - PromptBuilder: 提示词构建，支持 Jinja2 模板
    - ReplicationAgent: 论文复现 Agent
"""
from agent.session import (
    Session,
    SessionManager,
    SessionStatus,
    RoundResult,
)
from agent.prompt.prompt_builder import PromptBuilder
from agent.replication_agent import ReplicationAgent


__all__ = [
    # Session 管理
    "Session",
    "SessionManager",
    "SessionStatus",
    "RoundResult",
    # 提示词构建
    "PromptBuilder",
    # Agent
    "ReplicationAgent",
]
