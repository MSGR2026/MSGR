"""
挂载路径构建器

根据任务配置生成 Singularity 挂载路径
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..configs import TaskConfig


class MountBuilder:
    """
    挂载路径构建器
    
    根据任务配置和运行时参数生成挂载路径列表
    """
    
    def __init__(self, shared_root: str):
        """
        Args:
            shared_root: 共享存储根目录
        """
        self.shared_root = Path(shared_root)
    
    def build(
        self,
        task_config: TaskConfig,
        model: str,
        dataset: str,
        algorithm_path: Path,
        hyperparameter_path: Path,
        log_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> List[str]:
        """
        构建挂载路径列表
        
        Args:
            task_config: 任务配置
            model: 模型名
            dataset: 数据集名
            algorithm_path: 算法文件路径
            hyperparameter_path: 超参数文件路径
            log_dir: 日志目录（可选）
            run_id: 运行 ID（可选）
            
        Returns:
            List[str]: 挂载路径列表，格式 ["host:container", ...]
        """
        binds = []
        
        # 变量替换字典
        variables = {
            "model": model,
            "model_lower": model.lower(),
            "dataset": dataset,
            "algorithm_path": str(algorithm_path),
            "hyperparameter_path": str(hyperparameter_path),
            "run_id": run_id or "default",
            "shared_root": str(self.shared_root),
        }
        
        # 处理每个挂载配置
        for name, mount in task_config.mount.items():
            host = self._expand_path(mount.host, variables)
            container = self._expand_path(mount.container, variables)
            
            # 确保主机路径是绝对路径
            if not host.startswith("/"):
                host = str(self.shared_root / host)
            
            # 如果是目录挂载（container 路径以 / 结尾或不含 . 扩展名），确保目录存在
            host_path = Path(host)
            if not host_path.exists():
                # 如果是日志类目录，自动创建
                if "log" in name.lower() or "output" in name.lower() or "tensorboard" in name.lower():
                    host_path.mkdir(parents=True, exist_ok=True)
            
            binds.append(f"{host}:{container}")
        
        # 添加日志挂载（如果配置中没有已经挂载 /logs）
        if log_dir:
            # 检查是否已经有 /logs 挂载
            has_logs_mount = any(":/logs" in bind for bind in binds)
            if not has_logs_mount:
                binds.append(f"{log_dir}:/logs")
        
        return binds
    
    def _expand_path(self, path: str, variables: Dict[str, str]) -> str:
        """
        展开路径中的变量
        
        Args:
            path: 包含变量的路径（如 "/app/src/models/{model_lower}.py"）
            variables: 变量字典
            
        Returns:
            str: 展开后的路径
        """
        result = path
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
    
    def get_container_paths(
        self,
        task_config: TaskConfig,
        model: str,
        dataset: str,
    ) -> Dict[str, str]:
        """
        获取容器内路径
        
        Args:
            task_config: 任务配置
            model: 模型名
            dataset: 数据集名
            
        Returns:
            Dict[str, str]: 容器内路径字典 {name: container_path}
        """
        variables = {
            "model": model,
            "model_lower": model.lower(),
            "dataset": dataset,
        }
        
        paths = {}
        for name, mount in task_config.mount.items():
            paths[name] = self._expand_path(mount.container, variables)
        
        return paths


def build_bind_paths(
    task_config: TaskConfig,
    shared_root: str,
    model: str,
    dataset: str,
    algorithm_path: Path,
    hyperparameter_path: Path,
    log_dir: Optional[Path] = None,
) -> List[str]:
    """
    便捷函数：构建挂载路径
    
    Args:
        task_config: 任务配置
        shared_root: 共享存储根目录
        model: 模型名
        dataset: 数据集名
        algorithm_path: 算法文件路径
        hyperparameter_path: 超参数文件路径
        log_dir: 日志目录
        
    Returns:
        List[str]: 挂载路径列表
    """
    builder = MountBuilder(shared_root)
    return builder.build(
        task_config=task_config,
        model=model,
        dataset=dataset,
        algorithm_path=algorithm_path,
        hyperparameter_path=hyperparameter_path,
        log_dir=log_dir,
    )

