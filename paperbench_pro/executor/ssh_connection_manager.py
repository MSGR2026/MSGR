"""
SSH Connection Manager - 使用 ControlMaster 复用 SSH 连接

解决并发 SSH 连接过多导致的 "ssh_exchange_identification: Connection closed by remote host" 问题。

Usage:
    manager = SSHConnectionManager()
    
    # 在任务开始前建立连接
    manager.ensure_connections(["gpu4", "gpu5"])
    
    # 任务执行时使用 get_ssh_options()
    ssh_options = manager.get_ssh_options()
    ssh_cmd = ["ssh"] + ssh_options + [node, command]
    
    # 任务结束后清理
    manager.close_all()
"""
from __future__ import annotations

import atexit
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


class SSHConnectionManager:
    """
    SSH 连接管理器，使用 ControlMaster 实现连接复用
    
    原理：
    1. 为每个节点建立一个 Master 连接（后台运行）
    2. 后续 SSH 连接通过 ControlPath 复用 Master 连接
    3. 避免每个任务都建立新连接，绕过 sshd 的 MaxStartups 限制
    """
    
    _instance: Optional["SSHConnectionManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "SSHConnectionManager":
        """单例模式，确保全局只有一个连接管理器"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._initialized = True
        self._connections: Dict[str, subprocess.Popen] = {}
        self._conn_lock = threading.Lock()
        
        # 创建 socket 目录
        self._socket_dir = Path(tempfile.gettempdir()) / f"ssh_control_{os.getpid()}"
        self._socket_dir.mkdir(parents=True, exist_ok=True)
        
        # 注册退出清理
        atexit.register(self.close_all)
    
    @property
    def socket_dir(self) -> Path:
        return self._socket_dir
    
    def get_socket_path(self, node: str) -> str:
        """获取节点的 socket 路径"""
        return str(self._socket_dir / f"{node}.sock")
    
    def get_ssh_options(self, node: Optional[str] = None) -> List[str]:
        """
        获取 SSH 选项，用于复用 Master 连接
        
        Args:
            node: 节点名称（可选，用于生成特定节点的 socket 路径）
            
        Returns:
            SSH 命令行选项列表
        """
        if node:
            socket_path = self.get_socket_path(node)
        else:
            # 使用模板，SSH 会自动替换
            socket_path = str(self._socket_dir / "%h.sock")
        
        return [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={socket_path}",
            "-o", "ControlPersist=300",  # 空闲 5 分钟后关闭
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=no",  # 避免首次连接交互
        ]
    
    def is_connected(self, node: str) -> bool:
        """检查节点是否已有 Master 连接"""
        socket_path = self.get_socket_path(node)
        if not os.path.exists(socket_path):
            return False
        
        # 使用 ssh -O check 验证连接是否有效
        try:
            result = subprocess.run(
                ["ssh", "-O", "check", "-o", f"ControlPath={socket_path}", node],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def establish_connection(self, node: str, timeout: int = 30) -> bool:
        """
        建立到节点的 Master 连接
        
        Args:
            node: 节点名称
            timeout: 连接超时（秒）
            
        Returns:
            是否成功建立连接
        """
        with self._conn_lock:
            # 检查是否已连接
            if self.is_connected(node):
                return True
            
            socket_path = self.get_socket_path(node)
            
            # 清理可能存在的旧 socket
            if os.path.exists(socket_path):
                try:
                    os.remove(socket_path)
                except OSError:
                    pass
            
            # 建立 Master 连接（后台运行，不执行任何命令）
            ssh_cmd = [
                "ssh",
                "-o", "ControlMaster=yes",
                "-o", f"ControlPath={socket_path}",
                "-o", "ControlPersist=yes",
                "-o", f"ConnectTimeout={timeout}",
                "-o", "ServerAliveInterval=60",
                "-o", "ServerAliveCountMax=3",
                "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes",
                "-fN",  # 后台运行，不执行命令
                node,
            ]
            
            try:
                process = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                
                # 等待连接建立
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.is_connected(node):
                        self._connections[node] = process
                        return True
                    
                    # 检查进程是否已退出（失败）
                    if process.poll() is not None:
                        stderr = process.stderr.read().decode() if process.stderr else ""
                        print(f"[SSHManager] Failed to connect to {node}: {stderr}")
                        return False
                    
                    time.sleep(0.2)
                
                # 超时
                process.kill()
                print(f"[SSHManager] Connection to {node} timed out")
                return False
                
            except Exception as e:
                print(f"[SSHManager] Error connecting to {node}: {e}")
                return False
    
    def ensure_connections(
        self, 
        nodes: List[str], 
        timeout: int = 30,
        retry: int = 2,
        delay_between: float = 0.5,
    ) -> Dict[str, bool]:
        """
        确保到所有节点的连接已建立
        
        Args:
            nodes: 节点列表
            timeout: 每个连接的超时时间
            retry: 失败重试次数
            delay_between: 节点间连接延迟（避免同时发起太多连接）
            
        Returns:
            每个节点的连接状态 {node: success}
        """
        results = {}
        
        for i, node in enumerate(nodes):
            if i > 0 and delay_between > 0:
                time.sleep(delay_between)
            
            success = False
            for attempt in range(retry + 1):
                if self.establish_connection(node, timeout):
                    success = True
                    break
                if attempt < retry:
                    print(f"[SSHManager] Retrying connection to {node} ({attempt + 1}/{retry})")
                    time.sleep(1)
            
            results[node] = success
            if success:
                print(f"[SSHManager] ✓ Connected to {node}")
            else:
                print(f"[SSHManager] ✗ Failed to connect to {node}")
        
        return results
    
    def close_connection(self, node: str) -> None:
        """关闭到节点的 Master 连接"""
        with self._conn_lock:
            socket_path = self.get_socket_path(node)
            
            # 发送退出信号
            try:
                subprocess.run(
                    ["ssh", "-O", "exit", "-o", f"ControlPath={socket_path}", node],
                    capture_output=True,
                    timeout=5,
                )
            except (subprocess.TimeoutExpired, Exception):
                pass
            
            # 清理 socket 文件
            if os.path.exists(socket_path):
                try:
                    os.remove(socket_path)
                except OSError:
                    pass
            
            # 清理进程记录
            if node in self._connections:
                try:
                    self._connections[node].kill()
                except Exception:
                    pass
                del self._connections[node]
    
    def close_all(self) -> None:
        """关闭所有 Master 连接"""
        with self._conn_lock:
            nodes = list(self._connections.keys())
        
        for node in nodes:
            self.close_connection(node)
        
        # 清理 socket 目录
        try:
            if self._socket_dir.exists():
                for f in self._socket_dir.iterdir():
                    try:
                        f.unlink()
                    except OSError:
                        pass
                self._socket_dir.rmdir()
        except OSError:
            pass
    
    def status(self) -> Dict[str, bool]:
        """获取所有节点的连接状态"""
        with self._conn_lock:
            nodes = list(self._connections.keys())
        return {node: self.is_connected(node) for node in nodes}


# 全局单例
_manager: Optional[SSHConnectionManager] = None


def get_ssh_manager() -> SSHConnectionManager:
    """获取全局 SSH 连接管理器"""
    global _manager
    if _manager is None:
        _manager = SSHConnectionManager()
    return _manager

