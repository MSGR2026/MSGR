from __future__ import annotations

import os
import selectors
import shlex
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

from .base import CommandResult, ExecutorBase
from .ssh_connection_manager import get_ssh_manager


class SSHSingularityExecutor(ExecutorBase):
    def __init__(
        self,
        sif_path: str,
        bind_paths: List[str],
        node: str,
        gpu_id: Optional[int] = None,
        singularity_exe: str = "singularity",
        containall: bool = False,
        conda_env: Optional[str] = None,
        use_control_master: bool = True,  # 是否使用 SSH 连接复用
    ) -> None:
        self.sif_path = sif_path
        self.bind_paths = bind_paths
        self.node = node
        self.gpu_id = gpu_id
        self.singularity_exe = singularity_exe
        self.containall = containall
        self.conda_env = conda_env
        self.use_control_master = use_control_master

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
        start_time = time.time()
        env = dict(env or {})
        cancelled = False  # 追踪是否被 fast fail 取消

        if self.gpu_id is not None:
            env.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu_id))

        singularity_cmd = self._build_singularity_command(command, work_dir, env)
        
        # 构建 SSH 命令，支持 ControlMaster 连接复用
        if self.use_control_master:
            ssh_manager = get_ssh_manager()
            ssh_options = ssh_manager.get_ssh_options(self.node)
            ssh_cmd = ["ssh"] + ssh_options + [self.node, singularity_cmd]
        else:
            ssh_cmd = ["ssh", self.node, singularity_cmd]

        process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        timed_out = False

        selector = selectors.DefaultSelector()
        if process.stdout:
            selector.register(process.stdout, selectors.EVENT_READ)
        if process.stderr:
            selector.register(process.stderr, selectors.EVENT_READ)

        while True:
            # 检查 fast fail 停止信号
            if stop_event and stop_event.is_set():
                cancelled = True
                process.kill()
                break
            
            if timeout and (time.time() - start_time) > timeout:
                timed_out = True
                process.kill()
                break

            if process.poll() is not None and not selector.get_map():
                break

            for key, _ in selector.select(timeout=0.2):
                line = key.fileobj.readline()
                if not line:
                    selector.unregister(key.fileobj)
                    continue
                line = line.rstrip("\n")
                if key.fileobj is process.stdout:
                    stdout_lines.append(line)
                    if line_callback:
                        line_callback(line, False)
                    if streaming:
                        print(f"[paperbench] {line}")
                else:
                    stderr_lines.append(line)
                    if line_callback:
                        line_callback(line, True)
                    if streaming:
                        print(f"[paperbench][stderr] {line}")

            if process.poll() is not None and not selector.get_map():
                break

        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

        return_code = process.wait() if process.returncode is None else process.returncode
        duration = time.time() - start_time

        return CommandResult(
            return_code=return_code,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            duration=duration,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    def _build_singularity_command(
        self,
        command: List[str],
        work_dir: str,
        env: Dict[str, str],
    ) -> str:
        cmd = [self.singularity_exe, "exec"]
        if self.containall:
            cmd.append("--containall")
        cmd.append("--nv")
        if work_dir:
            cmd.extend(["--pwd", work_dir])
        for bind in self.bind_paths:
            cmd.extend(["-B", bind])
        cmd.append(self.sif_path)

        cmd_str = " ".join(shlex.quote(part) for part in command)
        export_parts = []
        if self.conda_env:
            export_parts.append(f"export PATH=/opt/conda/envs/{self.conda_env}/bin:$PATH;")
        for key, value in env.items():
            if key == "PATH":
                continue
            export_parts.append(f"export {key}={shlex.quote(str(value))};")
        export_cmd = " ".join(export_parts)
        inner = f"{export_cmd} {cmd_str}".strip()
        cmd.extend(["bash", "-lc", shlex.quote(inner)])

        return " ".join(cmd)
