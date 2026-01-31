"""
PaperBenchPro Configs Module

配置加载和管理
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ==============================================================================
# 数据类定义
# ==============================================================================

@dataclass
class MountConfig:
    """挂载配置"""
    host: str
    container: str


@dataclass
class ExecutionConfig:
    """执行配置"""
    container: str
    entry: str
    workdir: str = "/app"
    timeout: int = 3600


@dataclass
class DataConfig:
    """数据配置"""
    datasets: List[str] = field(default_factory=list)
    root: str = ""


@dataclass
class EvalConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=list)
    primary_metric: str = ""
    higher_is_better: bool = True


@dataclass
class TaskConfig:
    """任务配置"""
    domain: str
    task: str
    framework: str = ""
    description: str = ""
    
    execution: Optional[ExecutionConfig] = None
    mount: Dict[str, MountConfig] = field(default_factory=dict)
    data: Optional[DataConfig] = None
    eval: Optional[EvalConfig] = None
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SSHConfig:
    """SSH 配置"""
    nodes: List[str] = field(default_factory=list)
    user: str = ""
    timeout: int = 30


@dataclass
class SingularityConfig:
    """Singularity 配置"""
    executable: str = "singularity"
    options: List[str] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """调度器配置"""
    gpus_per_node: int = 8
    max_per_gpu: int = 2
    max_concurrent: int = 16


@dataclass
class ClusterConfig:
    """集群配置"""
    shared_root: str = ""
    ssh: Optional[SSHConfig] = None
    singularity: Optional[SingularityConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    timeout: int = 3600


# ==============================================================================
# 批量运行配置
# ==============================================================================

@dataclass
class AlgorithmConfig:
    """
    单个算法的配置
    
    用于批量运行多个算法时，描述每个算法的信息。
    
    Attributes:
        name: 算法名称（如 "BM3", "LATTICE"）
        algorithm_path: 算法实现文件路径
        hyperparameter_path: 超参数配置文件路径
    """
    name: str
    algorithm_path: Path
    hyperparameter_path: Path
    
    def __post_init__(self):
        self.algorithm_path = Path(self.algorithm_path)
        self.hyperparameter_path = Path(self.hyperparameter_path)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlgorithmConfig":
        """从字典创建配置"""
        return cls(
            name=data["name"],
            algorithm_path=Path(data["algorithm_path"]),
            hyperparameter_path=Path(data["hyperparameter_path"]),
        )


@dataclass
class BatchRunConfig:
    """
    批量运行配置
    
    用于同时运行多个算法的场景（如批量复现、批量评测）。
    支持按算法粒度的 fail-fast 控制。
    
    Attributes:
        domain: 领域（如 Recsys, TimeSeries）
        task: 任务（如 MultiModal, LongTermForecast）
        algorithms: 算法配置列表
        datasets: 数据集列表（None 表示使用配置中的默认数据集）
        expand_hyperparameters: 是否展开超参数网格
        max_concurrent: 最大并发任务数
        timeout: 单任务超时时间（秒）
        fail_fast_per_algorithm: 按算法粒度快速失败
        output_dir: 输出目录
        save_logs: 是否保存日志
        quiet: 静默模式
        nodes: 覆盖节点列表
        gpus_per_node: 覆盖每节点 GPU 数
    """
    # 任务标识
    domain: str
    task: str
    
    # 算法配置
    algorithms: List[AlgorithmConfig]
    
    # 数据集
    datasets: Optional[List[str]] = None
    
    # 调度选项
    expand_hyperparameters: bool = True
    max_concurrent: int = 16
    timeout: int = 3600
    fail_fast_per_algorithm: bool = True
    
    # 输出选项
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_logs: bool = True
    quiet: bool = True
    
    # 覆盖配置
    nodes: Optional[List[str]] = None
    gpus_per_node: Optional[int] = None
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        # 确保 algorithms 是 AlgorithmConfig 对象列表
        self.algorithms = [
            AlgorithmConfig.from_dict(a) if isinstance(a, dict) else a
            for a in self.algorithms
        ]
    
    @property
    def algorithm_names(self) -> List[str]:
        """获取所有算法名称"""
        return [a.name for a in self.algorithms]


# ==============================================================================
# 配置加载器
# ==============================================================================

def _load_yaml(path: Path) -> dict:
    """加载 YAML 文件"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_task_config(domain: str, task: str, config_dir: Path = None) -> TaskConfig:
    """
    加载任务配置
    
    Args:
        domain: 领域（如 Recsys）
        task: 任务（如 MultiModal）
        config_dir: 配置目录（默认 paperbench_pro/configs）
        
    Returns:
        TaskConfig: 任务配置
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    config_path = config_dir / "tasks" / domain / f"{task}.yaml"
    data = _load_yaml(config_path)
    
    if not data:
        raise FileNotFoundError(f"Task config not found: {config_path}")
    
    # 解析配置
    task_data = data.get("task", {})
    
    # 执行配置
    exec_data = data.get("execution", {})
    execution = ExecutionConfig(**exec_data) if exec_data else None
    
    # 挂载配置
    mount_data = data.get("mount", {})
    mount = {k: MountConfig(**v) for k, v in mount_data.items()}
    
    # 数据配置
    data_cfg = data.get("data", {})
    data_config = DataConfig(**data_cfg) if data_cfg else None
    
    # 评估配置
    eval_data = data.get("eval", {})
    eval_config = EvalConfig(**eval_data) if eval_data else None
    
    return TaskConfig(
        domain=task_data.get("domain", domain),
        task=task_data.get("task", task),
        framework=task_data.get("framework", ""),
        description=task_data.get("description", ""),
        execution=execution,
        mount=mount,
        data=data_config,
        eval=eval_config,
        args=data.get("args", {}),
    )


def load_cluster_config(config_dir: Path = None) -> ClusterConfig:
    """
    加载集群配置
    
    Args:
        config_dir: 配置目录
        
    Returns:
        ClusterConfig: 集群配置
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    config_path = config_dir / "cluster.yaml"
    data = _load_yaml(config_path)
    
    # 如果没有 cluster.yaml，使用默认值
    if not data:
        return ClusterConfig(
            shared_root=".",
            ssh=SSHConfig(nodes=["gpu3", "gpu4"]),
            singularity=SingularityConfig(
                executable="singularity"
            ),
            scheduler=SchedulerConfig(),
        )
    
    # SSH 配置
    ssh_data = data.get("ssh", {})
    ssh = SSHConfig(**ssh_data) if ssh_data else None
    
    # Singularity 配置
    sing_data = data.get("singularity", {})
    singularity = SingularityConfig(**sing_data) if sing_data else None
    
    # 调度器配置
    sched_data = data.get("scheduler", {})
    scheduler = SchedulerConfig(**sched_data) if sched_data else None
    
    return ClusterConfig(
        shared_root=data.get("shared_root", ""),
        ssh=ssh,
        singularity=singularity,
        scheduler=scheduler,
        timeout=data.get("timeout", 3600),
    )


def list_tasks(config_dir: Path = None) -> List[str]:
    """
    列出所有可用任务
    
    Returns:
        List[str]: 任务列表（格式：Domain/Task）
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    tasks_dir = config_dir / "tasks"
    if not tasks_dir.exists():
        return []
    
    tasks = []
    for domain_dir in tasks_dir.iterdir():
        if domain_dir.is_dir():
            domain = domain_dir.name
            for task_file in domain_dir.glob("*.yaml"):
                task = task_file.stem
                tasks.append(f"{domain}/{task}")
    
    return sorted(tasks)


__all__ = [
    # 任务配置
    "TaskConfig",
    "ClusterConfig",
    "ExecutionConfig",
    "MountConfig",
    "DataConfig",
    "EvalConfig",
    "SSHConfig",
    "SingularityConfig",
    "SchedulerConfig",
    # 批量运行配置
    "AlgorithmConfig",
    "BatchRunConfig",
    # 加载器
    "load_task_config",
    "load_cluster_config",
    "list_tasks",
]

