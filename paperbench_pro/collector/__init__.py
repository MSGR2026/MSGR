"""
PaperBench-Pro Collector 模块

简化设计：
- ResultTable: 追踪多数据集最佳结果 + 实时显示表格
- parse_output(): 解析日志输出，返回指标字典
- save_results(): 保存结果到文件
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .result_table import (
    ResultTable, DatasetResult, BestResult,
    BatchResultTable, AlgorithmProgress,
)
from .recbole_collector import RecBoleCollector
from .mmrec_collector import MMRecCollector


# ==============================================================================
# 日志解析
# ==============================================================================

def parse_output(
    stdout: str,
    domain: str = "Recsys",
    task: str = "MultiModal",
) -> Dict[str, float]:
    """
    解析日志输出，返回指标字典
    
    Args:
        stdout: 标准输出内容
        domain: 领域 (Recsys, TimeSeries, ...)
        task: 任务 (MultiModal, General, ...)
    
    Returns:
        Dict[str, float]: 指标字典，如 {"recall@20": 0.0677, "ndcg@10": 0.0312}
    """
    # 根据 domain/task 选择解析器
    if domain == "Recsys":
        if task in ("MultiModal",):
            collector = MMRecCollector.from_string(stdout)
        else:
            # General, Sequential, ContextAware, KnowledgeBased 都用 RecBole
            collector = RecBoleCollector.from_string(stdout)
        return collector.get_metrics()
    
    # TODO: TimeSeries, GraphLearning 等
    return {}


# ==============================================================================
# 结果数据结构
# ==============================================================================

@dataclass
class TaskResult:
    """
    单个任务的执行结果
    
    Attributes:
        algorithm: 算法名称（批量运行时使用）
        dataset: 数据集名称
        hp_name: 超参数配置名称
        metrics: 评估指标字典
        success: 是否成功
        duration: 执行时长（秒）
        error: 错误信息
        raw_stdout: 完整的 stdout 输出（用于调试）
        raw_stderr: 完整的 stderr 输出（用于调试）
        node: 执行节点
        gpu_id: GPU ID
    """
    # 任务标识
    algorithm: str = ""  # 批量运行时使用，单算法运行时为空
    dataset: str = ""
    hp_name: str = ""
    
    # 执行结果
    metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    duration: float = 0.0
    error: Optional[str] = None
    
    # 调试信息
    raw_stdout: Optional[str] = None
    raw_stderr: Optional[str] = None
    
    # 执行环境
    node: str = ""
    gpu_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（不包含原始输出以减小体积）"""
        d = asdict(self)
        # 如果成功，不保存原始输出
        if self.success:
            d.pop('raw_stdout', None)
            d.pop('raw_stderr', None)
        return d
    
    def get_full_error(self) -> str:
        """获取完整的错误信息（包含 stdout 和 stderr）"""
        parts = []
        if self.error:
            parts.append(f"Error: {self.error}")
        if self.raw_stderr:
            parts.append(f"Stderr:\n{self.raw_stderr}")
        if self.raw_stdout:
            parts.append(f"Stdout:\n{self.raw_stdout}")
        return "\n".join(parts) if parts else "Unknown error"


@dataclass  
class RunSummary:
    """运行摘要"""
    domain: str
    task: str
    model: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_tasks: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    results: List[TaskResult] = field(default_factory=list)
    best_per_dataset: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_result(self, result: TaskResult):
        """添加任务结果"""
        self.results.append(result)
        self.completed += 1
        if result.success:
            self.success += 1
        else:
            self.failed += 1
    
    def finalize(self, best_results: Dict[str, DatasetResult]):
        """完成并设置最佳结果"""
        self.end_time = datetime.now().isoformat()
        for ds, result in best_results.items():
            self.best_per_dataset[ds] = {
                "metrics": result.metrics,
                "hp": result.best_hp,
            }
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


# ==============================================================================
# 结果保存
# ==============================================================================

def save_results(
    summary: RunSummary,
    output_dir: Path,
    filename: str = None,
) -> Path:
    """
    保存结果到 JSON 文件
    
    Args:
        summary: 运行摘要
        output_dir: 输出目录
        filename: 文件名（默认自动生成）
    
    Returns:
        Path: 保存的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{summary.model}_{timestamp}.json"
    
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    
    return filepath


def generate_report(summary: RunSummary) -> str:
    """
    生成文本报告
    
    Args:
        summary: 运行摘要
    
    Returns:
        str: 格式化的报告文本
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {summary.domain}/{summary.task} - {summary.model}")
    lines.append("=" * 70)
    lines.append(f"  Total: {summary.completed}/{summary.total_tasks}  |  "
                 f"Success: {summary.success}  |  Failed: {summary.failed}")
    lines.append("-" * 70)
    
    if summary.best_per_dataset:
        lines.append("  Best Results per Dataset:")
        for ds, result in summary.best_per_dataset.items():
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items())
            lines.append(f"    {ds}: {metrics_str}  [HP: {result['hp']}]")
    
    lines.append("=" * 70)
    lines.append("")
    
    return "\n".join(lines)


# ==============================================================================
# 批量运行摘要
# ==============================================================================

@dataclass
class BatchRunSummary:
    """
    批量运行摘要
    
    用于多算法并行运行的结果汇总。
    """
    domain: str
    task: str
    algorithms: List[str]
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_tasks: int = 0
    completed: int = 0
    
    # 按算法分组的结果
    results_by_algorithm: Dict[str, List[TaskResult]] = field(default_factory=dict)
    best_by_algorithm: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    fatal_by_algorithm: Dict[str, Optional[TaskResult]] = field(default_factory=dict)
    
    def add_result(self, result: TaskResult):
        """添加任务结果"""
        algo = result.algorithm
        if algo not in self.results_by_algorithm:
            self.results_by_algorithm[algo] = []
        self.results_by_algorithm[algo].append(result)
        self.completed += 1
    
    def set_fatal(self, algorithm: str, result: TaskResult):
        """设置算法的致命错误"""
        self.fatal_by_algorithm[algorithm] = result
    
    def finalize(self, best_results: Dict[str, Dict[str, DatasetResult]]):
        """
        完成摘要
        
        Args:
            best_results: 按算法分组的最佳结果
                {algorithm: {dataset: DatasetResult}}
        """
        self.end_time = datetime.now().isoformat()
        for algo, ds_results in best_results.items():
            self.best_by_algorithm[algo] = {
                ds: {
                    "metrics": result.metrics,
                    "hp": result.best_hp,
                }
                for ds, result in ds_results.items()
            }
    
    def get_algorithm_stats(self, algorithm: str) -> Dict[str, int]:
        """获取算法的统计信息"""
        results = self.results_by_algorithm.get(algorithm, [])
        success = sum(1 for r in results if r.success)
        return {
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "domain": self.domain,
            "task": self.task,
            "algorithms": self.algorithms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "results_by_algorithm": {
                algo: [r.to_dict() for r in results]
                for algo, results in self.results_by_algorithm.items()
            },
            "best_by_algorithm": self.best_by_algorithm,
            "fatal_by_algorithm": {
                algo: r.to_dict() if r else None
                for algo, r in self.fatal_by_algorithm.items()
            },
        }


def generate_batch_report(summary: BatchRunSummary) -> str:
    """
    生成批量运行报告
    
    Args:
        summary: 批量运行摘要
    
    Returns:
        str: 格式化的报告文本
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  {summary.domain}/{summary.task} - Batch Run Summary")
    lines.append("=" * 80)
    lines.append(f"  Algorithms: {len(summary.algorithms)}  |  "
                 f"Total Tasks: {summary.completed}/{summary.total_tasks}")
    lines.append("-" * 80)
    
    for algo in summary.algorithms:
        stats = summary.get_algorithm_stats(algo)
        fatal = summary.fatal_by_algorithm.get(algo)
        
        # 算法状态标记
        if fatal:
            status_icon = "✗ STOPPED"
            status_color = "\033[31m"  # 红色
        elif stats["failed"] > 0:
            status_icon = "⚠ PARTIAL"
            status_color = "\033[33m"  # 黄色
        else:
            status_icon = "✓ DONE"
            status_color = "\033[32m"  # 绿色
        
        lines.append(f"\n  {status_color}[{status_icon}]\033[0m {algo}")
        lines.append(f"    Tasks: {stats['total']} (success: {stats['success']}, failed: {stats['failed']})")
        
        if fatal:
            lines.append(f"    Fatal Error: {fatal.error}")
        
        # 显示最佳结果
        if algo in summary.best_by_algorithm:
            best = summary.best_by_algorithm[algo]
            if best:
                lines.append(f"    Best Results:")
                for ds, info in best.items():
                    if info["metrics"]:
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in info["metrics"].items())
                        lines.append(f"      {ds}: {metrics_str}  [HP: {info['hp']}]")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    return "\n".join(lines)


def save_batch_results(
    summary: BatchRunSummary,
    output_dir: Path,
    filename: str = None,
) -> Path:
    """
    保存批量运行结果到 JSON 文件
    
    Args:
        summary: 批量运行摘要
        output_dir: 输出目录
        filename: 文件名（默认自动生成）
    
    Returns:
        Path: 保存的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_names = "_".join(summary.algorithms[:3])  # 最多取前3个算法名
        if len(summary.algorithms) > 3:
            algo_names += f"_+{len(summary.algorithms) - 3}"
        filename = f"batch_{algo_names}_{timestamp}.json"
    
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    
    return filepath


# ==============================================================================
# 导出
# ==============================================================================

__all__ = [
    # 结果表格
    "ResultTable",
    "DatasetResult",
    "BestResult",
    # 批量结果表格
    "BatchResultTable",
    "AlgorithmProgress",
    # 日志解析
    "parse_output",
    "RecBoleCollector",
    "MMRecCollector",
    # 结果数据结构
    "TaskResult",
    "RunSummary",
    "BatchRunSummary",
    # 结果保存与报告
    "save_results",
    "generate_report",
    "save_batch_results",
    "generate_batch_report",
]
