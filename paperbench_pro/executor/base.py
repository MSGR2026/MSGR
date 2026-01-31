from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class CommandResult:
    return_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    cancelled: bool = False  # 是否被 fast fail 取消


class ExecutorBase(ABC):
    @abstractmethod
    def execute(
        self,
        command: List[str],
        work_dir: str,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        streaming: bool = False,
        line_callback: Optional[Callable[[str, bool], None]] = None,
        stop_event: Optional[threading.Event] = None,  # fast fail 停止信号
    ) -> CommandResult:
        raise NotImplementedError
