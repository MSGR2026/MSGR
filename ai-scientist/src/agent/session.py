"""
Session Management for Scientific Agent.

Provides Session state tracking, history recording, and termination condition checking.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionStatus(Enum):
    """Session status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"           # 首次成功执行（无错误）
    FAILED = "failed"             # Reached max_rounds without success
    ERROR = "error"               # Runtime error


@dataclass
class RoundResult:
    """Result of a single replication round."""
    
    round: int
    algorithm_path: str = ""
    hyperparameter_path: str = ""
    acc: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    feedback: str = ""
    duration: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw_paperbench_data: Optional[Dict[str, Any]] = None  # 保存原始 PaperBench 数据（用于提取 combo 信息）
    
    @property
    def success(self) -> bool:
        """Whether the round was successful (no error)."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "round": self.round,
            "algorithm_path": self.algorithm_path,
            "hyperparameter_path": self.hyperparameter_path,
            "acc": self.acc,
            "metrics": self.metrics,
            "error": self.error,
            "feedback": self.feedback,
            "duration": self.duration,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        # raw_paperbench_data 不序列化到 session.json（太大）
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoundResult":
        """Create from dictionary."""
        return cls(
            round=d.get("round", 0),
            algorithm_path=d.get("algorithm_path", ""),
            hyperparameter_path=d.get("hyperparameter_path", ""),
            acc=d.get("acc"),
            metrics=d.get("metrics", {}),
            error=d.get("error"),
            feedback=d.get("feedback", ""),
            duration=d.get("duration", 0.0),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )
    
    @classmethod
    def from_paperbench(
        cls,
        data: Dict[str, Any],
        round_num: int,
        algorithm_path: str,
        hyperparameter_path: str,
        primary_metric: str = "recall@20",
        duration: float = 0.0,
    ) -> "RoundResult":
        """
        从 PaperBench-Pro 返回的结果创建 RoundResult。
        
        Args:
            data: PaperBench-Pro 返回的结果字典
            round_num: 当前轮次
            algorithm_path: 算法文件路径
            hyperparameter_path: 超参数文件路径
            primary_metric: 主要评测指标
            duration: 执行时长
        
        Returns:
            RoundResult 实例
        """
        # 检查是否有致命错误
        fatal_error = data.get("fatal_error")
        if fatal_error:
            # 从 stdout/stderr 构建完整错误信息
            stdout = fatal_error.get("stdout") or ""
            stderr = fatal_error.get("stderr") or ""
            
            # 构建错误信息：优先显示 stderr，然后是 stdout 的最后部分
            error_parts = []
            if stderr:
                error_parts.append(f"[stderr]\n{stderr}")
            if stdout:
                # stdout 可能很长，只取最后 2000 字符
                stdout_tail = stdout[-2000:] if len(stdout) > 2000 else stdout
                error_parts.append(f"[stdout]\n{stdout_tail}")
            
            error_msg = "\n".join(error_parts) if error_parts else "Unknown error"
            
            return cls(
                round=round_num,
                algorithm_path=algorithm_path,
                hyperparameter_path=hyperparameter_path,
                error=error_msg,
                feedback=f"执行失败:\n{error_msg}",
                duration=duration,
            )
        
        # 检查是否有成功结果（兼容 success_count 和 success）
        success_count = data.get("success_count") or data.get("success", 0)
        if success_count == 0:
            return cls(
                round=round_num,
                algorithm_path=algorithm_path,
                hyperparameter_path=hyperparameter_path,
                error="No successful runs",
                feedback="所有任务均失败",
                duration=duration,
            )
        
        # 提取最佳指标
        best = data.get("best", {})
        best_metrics = best.get("metrics", {})
        
        # 如果没有 best 字段，尝试从 realtime_tracking 获取
        if not best_metrics:
            realtime = data.get("realtime_tracking", {})
            # 兼容两种格式：best_by_dataset 或 datasets
            datasets_info = realtime.get("best_by_dataset") or realtime.get("datasets", {})
            # 找到主指标最高的数据集
            best_value = 0.0
            for ds, ds_info in datasets_info.items():
                ds_metrics = ds_info.get("metrics", {})
                value = ds_metrics.get(primary_metric, 0.0)
                if value > best_value:
                    best_value = value
                    best_metrics = ds_metrics
        
        # 获取主指标值
        acc = best_metrics.get(primary_metric, 0.0)
        
        # 构建反馈信息（兼容 total 和 total_tasks）
        total = data.get("total") or data.get("total_tasks", 0)
        feedback_parts = []
        feedback_parts.append(f"成功: {success_count}/{total}")
        if acc > 0:
            feedback_parts.append(f"{primary_metric}={acc:.4f}")
        feedback = ", ".join(feedback_parts)
        
        return cls(
            round=round_num,
            algorithm_path=algorithm_path,
            hyperparameter_path=hyperparameter_path,
            acc=acc,
            metrics=best_metrics,
            feedback=feedback,
            duration=duration,
        )


@dataclass
class Session:
    """Replication session state."""
    
    # Basic information
    session_id: str
    paper_id: str
    model_name: str
    domain: str
    task: str
    
    # Configuration
    max_rounds: int = 10
    
    # State
    current_round: int = 0
    status: SessionStatus = field(default=SessionStatus.PENDING)
    
    # History
    history: List[RoundResult] = field(default_factory=list)
    best_acc: float = 0.0
    best_round: int = 0
    
    # Timestamps
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    # Output path
    output_dir: str = ""
    
    def should_stop(self) -> bool:
        """
        Check if the session should stop.
        
        Stopping conditions:
        1. 最近一轮成功执行（无错误）→ SUCCESS
        2. 达到最大轮次 → FAILED
        
        Returns:
            True if should stop, False otherwise.
        """
        # 1. 最近一轮成功执行 → 停止
        if len(self.history) > 0 and self.history[-1].success:
            self.status = SessionStatus.SUCCESS
            return True
        
        # 2. 达到最大轮次 → 失败
        if self.current_round >= self.max_rounds:
            self.status = SessionStatus.FAILED
            return True
        
        return False
    
    def update_after_round(self, result: RoundResult) -> None:
        """
        Update session state after a round completes.
        
        Args:
            result: The round result.
        """
        self.history.append(result)
        
        # 更新最佳结果
        if result.success and result.acc is not None:
            if result.acc > self.best_acc:
                self.best_acc = result.acc
                self.best_round = result.round
        
        self.current_round += 1
    
    def start(self) -> None:
        """Mark session as started."""
        self.status = SessionStatus.RUNNING
        self.started_at = datetime.now()
    
    def finish(self) -> None:
        """Mark session as finished."""
        self.finished_at = datetime.now()
    
    def get_last_n_results(self, n: int = 3) -> List[RoundResult]:
        """Get the last N round results."""
        return self.history[-n:] if self.history else []
    
    def get_total_tokens(self) -> int:
        """Get total tokens used across all rounds."""
        return sum(r.total_tokens for r in self.history)
    
    def get_total_duration(self) -> float:
        """Get total duration across all rounds."""
        return sum(r.duration for r in self.history)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "paper_id": self.paper_id,
            "model_name": self.model_name,
            "domain": self.domain,
            "task": self.task,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "status": self.status.value,
            "history": [r.to_dict() for r in self.history],
            "best_acc": self.best_acc,
            "best_round": self.best_round,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "output_dir": self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        session = cls(
            session_id=d.get("session_id", ""),
            paper_id=d.get("paper_id", ""),
            model_name=d.get("model_name", ""),
            domain=d.get("domain", ""),
            task=d.get("task", ""),
            max_rounds=d.get("max_rounds", 10),
            current_round=d.get("current_round", 0),
            status=SessionStatus(d.get("status", "pending")),
            best_acc=d.get("best_acc", 0.0),
            best_round=d.get("best_round", 0),
            output_dir=d.get("output_dir", ""),
        )
        
        # Parse history
        session.history = [RoundResult.from_dict(r) for r in d.get("history", [])]
        
        # Parse timestamps
        if d.get("started_at"):
            session.started_at = datetime.fromisoformat(d["started_at"])
        if d.get("finished_at"):
            session.finished_at = datetime.fromisoformat(d["finished_at"])
        
        return session


class SessionManager:
    """Session manager for creating and tracking sessions."""
    
    def __init__(self, output_base: str = "ai-scientist/output"):
        """
        Initialize SessionManager.
        
        Args:
            output_base: Base directory for output files.
        """
        self.output_base = Path(output_base)
        self.sessions: Dict[str, Session] = {}
    
    def create_session(
        self,
        paper_id: str,
        model_name: str,
        domain: str,
        task: str,
        max_rounds: int = 10,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            paper_id: Target paper ID.
            model_name: Model name (alias).
            domain: Domain name.
            task: Task name.
            max_rounds: Maximum number of rounds.
            
        Returns:
            Created Session instance.
        """
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        session_id = f"{model_name}_{timestamp}_{unique_id}"
        
        # Create output directory path
        output_dir = str(self.output_base / domain / task / model_name)
        
        session = Session(
            session_id=session_id,
            paper_id=paper_id,
            model_name=model_name,
            domain=domain,
            task=task,
            max_rounds=max_rounds,
            output_dir=output_dir,
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def get_round_output_dir(self, session: Session) -> Path:
        """
        Get output directory for current round.
        
        Args:
            session: The session.
            
        Returns:
            Path to round output directory.
        """
        return Path(session.output_dir) / f"round_{session.current_round}"
    
    def ensure_round_output_dir(self, session: Session) -> Path:
        """
        Ensure round output directory exists.
        
        Args:
            session: The session.
            
        Returns:
            Path to round output directory.
        """
        output_dir = self.get_round_output_dir(session)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_session(self, session: Session, path: Optional[Path] = None) -> Path:
        """
        Save session state to JSON file.
        
        Args:
            session: The session to save.
            path: Optional custom path. If None, saves to session.output_dir/session.json.
            
        Returns:
            Path where session was saved.
        """
        import json
        
        if path is None:
            session_dir = Path(session.output_dir)
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / "session.json"
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        
        return path
    
    def load_session(self, path: Path) -> Session:
        """
        Load session from JSON file.
        
        Args:
            path: Path to session JSON file.
            
        Returns:
            Loaded Session instance.
        """
        import json
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        session = Session.from_dict(data)
        self.sessions[session.session_id] = session
        return session
    
    def list_sessions(self) -> List[Session]:
        """List all sessions."""
        return list(self.sessions.values())
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove session from manager.
        
        Args:
            session_id: Session ID to remove.
            
        Returns:
            True if removed, False if not found.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
