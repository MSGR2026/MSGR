"""
PaperBench-Pro 结果表格

美观的多任务结果展示表格，支持：
- 多数据集同时显示
- 多超参数组合搜索
- 实时更新最佳结果
- 进度条
- 改进提示
- 线程安全（支持多算法并行独立迭代）
"""
import sys
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class DatasetResult:
    """单个数据集的结果"""
    dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    best_hp: str = "-"
    status: str = "pending"  # pending, running, done, completed
    success_count: int = 0
    fail_count: int = 0
    
    def update(self, metrics: Dict[str, float], hp: str, mark_done: bool = True):
        """更新结果（如果更好）"""
        self.metrics = metrics
        self.best_hp = hp
        if mark_done:
            self.status = "done"


@dataclass
class BestResult:
    """最佳结果"""
    dataset: str
    metrics: Dict[str, float]
    hp_info: str


class ResultTable:
    """
    多任务结果表格
    
    使用示例：
        table = ResultTable(
            title="Current Best Test Metrics",
            datasets=["baby", "sports", "clothing"],
            metrics=["recall@20", "recall@10", "ndcg@20", "ndcg@10"],
            total_tasks=48
        )
        
        # 更新进度
        table.set_running("baby")
        table.update("baby", {"recall@20": 0.0677, ...}, hp="combo_2")
        table.complete_task()
        
        # 打印表格
        table.print()
    """
    
    def __init__(
        self,
        title: str = "Current Best Test Metrics",
        datasets: List[str] = None,
        metrics: List[str] = None,
        total_tasks: int = 0,
        primary_metric: str = "recall@20"
    ):
        self.title = title
        self.datasets = datasets or []
        self.metrics = metrics or ["recall@20", "recall@10", "ndcg@20", "ndcg@10"]
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.primary_metric = primary_metric
        
        # 每个数据集的结果
        self.results: Dict[str, DatasetResult] = {
            ds: DatasetResult(dataset=ds) for ds in self.datasets
        }
        
        # 记录上次值用于计算改进
        self.previous_values: Dict[str, float] = {}
        
        # 最近的改进信息
        self.last_improvement: Optional[str] = None
        self._last_print_lines: int = 0
        self._last_print_lines: int = 0
    
    def add_dataset(self, dataset: str):
        """添加数据集"""
        if dataset not in self.results:
            self.datasets.append(dataset)
            self.results[dataset] = DatasetResult(dataset=dataset)
    
    def set_running(self, dataset: str):
        """设置数据集为运行中"""
        if dataset in self.results:
            self.results[dataset].status = "running"
    
    def update(
        self,
        dataset: str,
        metrics: Dict[str, float],
        hp: str = "-",
        mark_done: bool = True,
    ) -> Optional[str]:
        """
        更新数据集结果
        
        Returns:
            改进信息字符串（如果有改进）
        """
        if dataset not in self.results:
            self.add_dataset(dataset)
        
        result = self.results[dataset]
        old_value = result.metrics.get(self.primary_metric, 0)
        new_value = metrics.get(self.primary_metric, 0)
        
        # 检查是否改进
        improvement = None
        if new_value > old_value:
            if old_value > 0:
                pct = (new_value - old_value) / old_value * 100
                improvement = f"✓ {dataset}: {self.primary_metric} improved {old_value:.4f} → {new_value:.4f} (+{pct:.1f}%) [{hp}]"
            else:
                improvement = f"✓ {dataset}: {self.primary_metric} = {new_value:.4f} [{hp}]"
            
            result.update(metrics, hp, mark_done=mark_done)
            self.last_improvement = improvement

        if mark_done:
            result.status = "done"
        return improvement
    
    def complete_task(self):
        """完成一个任务"""
        self.completed_tasks += 1
    
    def _format_value(self, value: Optional[float], status: str) -> str:
        """格式化值"""
        if value is None or value == 0:
            if status == "running":
                return "[running]"
            return "..."
        return f"{value:.4f}"
    
    def _progress_bar(self, width: int = 30) -> str:
        """生成进度条"""
        if self.total_tasks == 0:
            return " " * width
        
        filled = int(width * self.completed_tasks / self.total_tasks)
        bar = "█" * filled + "░" * (width - filled)
        pct = self.completed_tasks / self.total_tasks * 100
        return f"[{bar}]  {self.completed_tasks}/{self.total_tasks}  ({pct:4.0f}%)"
    
    def render(self) -> str:
        """渲染表格为字符串"""
        lines = []
        
        # 计算列宽
        ds_width = max(len(ds) for ds in self.datasets) if self.datasets else 8
        ds_width = max(ds_width, 8)
        metric_width = 10
        hp_width = 10
        
        # 表格总宽度
        total_width = ds_width + 3 + (metric_width + 3) * len(self.metrics) + hp_width + 2
        
        # 顶部边框
        lines.append("┌" + "─" * total_width + "┐")
        
        # 标题
        lines.append("│" + self.title.center(total_width) + "│")
        
        # 分隔线
        lines.append("├" + "─" * total_width + "┤")
        
        # 进度条
        progress = self._progress_bar(30)
        lines.append("│  Progress: " + progress.ljust(total_width - 12) + "│")
        
        # 分隔线
        lines.append("├" + "─" * total_width + "┤")
        
        # 表头
        header = f"│ {'Dataset':<{ds_width}} │"
        for m in self.metrics:
            header += f" {m:^{metric_width}} │"
        header += f" {'HP':^{hp_width}} │"
        lines.append(header)
        
        # 表头分隔线
        sep = "├" + "─" * (ds_width + 2) + "┼"
        for _ in self.metrics:
            sep += "─" * (metric_width + 2) + "┼"
        sep += "─" * (hp_width + 2) + "┤"
        lines.append(sep)
        
        # 数据行
        for ds in self.datasets:
            result = self.results[ds]
            row = f"│ {ds:<{ds_width}} │"
            
            for m in self.metrics:
                val = result.metrics.get(m)
                val_str = self._format_value(val, result.status)
                row += f" {val_str:^{metric_width}} │"
            
            row += f" {result.best_hp:^{hp_width}} │"
            lines.append(row)
        
        # 底部边框
        lines.append("└" + "─" * total_width + "┘")
        
        return "\n".join(lines)
    
    def print(self, clear: bool = False, consume_improvement: bool = False):
        """打印表格"""
        if not sys.stdout.isatty():
            clear = False
        if clear and self._last_print_lines:
            sys.stdout.write(f"\033[{self._last_print_lines}A\033[J")
        rendered = self.render()
        print(rendered)
        
        # 打印改进信息
        improvement_printed = False
        if self.last_improvement:
            print(f"\033[92m{self.last_improvement}\033[0m")  # 绿色
            improvement_printed = True
            if consume_improvement:
                self.last_improvement = None
        self._last_print_lines = rendered.count("\n") + 1 + (1 if improvement_printed else 0)
    
    def print_static(self):
        """静态打印（不清除）"""
        print(self.render())
    
    # ===========================================================================
    # interface.py 兼容方法
    # ===========================================================================
    
    def set_total_tasks(self, total: int):
        """设置总任务数"""
        self.total_tasks = total
    
    def set_target_datasets(self, datasets: List[str]):
        """设置目标数据集列表"""
        for ds in datasets:
            self.add_dataset(ds)
    
    def mark_running(self, dataset: str):
        """标记数据集为运行中（set_running 的别名）"""
        self.set_running(dataset)
    
    def mark_completed(self, dataset: str, success: bool = True):
        """标记任务完成"""
        self.complete_task()
        if dataset in self.results:
            result = self.results[dataset]
            if success:
                result.success_count += 1
            else:
                result.fail_count += 1

    def set_status(self, dataset: str, status: str) -> None:
        """设置数据集状态"""
        if dataset in self.results:
            self.results[dataset].status = status
    
    def display(self):
        """显示表格（print 的别名）"""
        self.print()
    
    def summary_dict(self) -> Dict[str, Any]:
        """返回摘要字典"""
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "datasets": {
                ds: {
                    "metrics": result.metrics,
                    "best_hp": result.best_hp,
                    "status": result.status,
                }
                for ds, result in self.results.items()
            }
        }
    
    def get_overall_best(self) -> Optional[BestResult]:
        """获取所有数据集中的最佳结果"""
        best = None
        best_value = 0.0
        
        for ds, result in self.results.items():
            value = result.metrics.get(self.primary_metric, 0)
            if value > best_value:
                best_value = value
                best = BestResult(
                    dataset=ds,
                    metrics=result.metrics.copy(),
                    hp_info=result.best_hp,
                )
        
        return best
    
    def get_best_results(self) -> Dict[str, DatasetResult]:
        """获取所有数据集的最佳结果"""
        return self.results.copy()


# ==============================================================================
# 批量结果表格（多算法）
# ==============================================================================

@dataclass
class AlgorithmProgress:
    """算法执行进度"""
    total: int = 0
    completed: int = 0
    success: int = 0
    failed: int = 0
    status: str = "pending"  # pending, running, done, failed, stopped
    current_round: int = 0   # 当前迭代轮次（用于多轮 Agent 场景）
    
    @property
    def is_done(self) -> bool:
        return self.completed >= self.total
    
    @property
    def success_rate(self) -> float:
        return self.success / self.completed if self.completed > 0 else 0.0


class BatchResultTable:
    """
    多算法结果表格（方案 B）
    
    支持同时追踪多个算法在多个数据集上的运行结果。
    采用合并表格布局：Algorithm × Dataset × Metrics
    
    Features:
        - 分组显示：按算法分组，每组内显示各数据集结果
        - 状态追踪：每个 (算法, 数据集) 组合的运行状态
        - 进度显示：总体进度 + 各算法进度
        - 停止标记：支持标记某算法被停止（fail-fast）
    
    Example:
        table = BatchResultTable(
            algorithms=["BM3", "LATTICE"],
            datasets=["baby", "sports"],
            metrics=["recall@20", "ndcg@20"],
            total_tasks=32,
            primary_metric="recall@20",
        )
        
        table.mark_running("BM3", "baby")
        table.update("BM3", "baby", {"recall@20": 0.0677}, "combo_2")
        table.mark_completed("BM3", "baby", success=True)
        
        table.display()
    """
    
    # 状态图标映射
    STATUS_ICONS = {
        "pending": "○ wait",
        "running": "◐ run ",
        "done": "✓ done",
        "failed": "✗ fail",
        "stopped": "⊘ stop",
    }
    
    def __init__(
        self,
        algorithms: List[str],
        datasets: List[str],
        metrics: List[str],
        total_tasks: int,
        primary_metric: str = "recall@20",
    ):
        self.algorithms = list(algorithms)  # 复制以避免外部修改
        self.datasets = list(datasets)
        self.metrics = list(metrics)
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.primary_metric = primary_metric
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        # 按 (算法, 数据集) 组织的结果
        self.results: Dict[str, Dict[str, DatasetResult]] = {
            algo: {ds: DatasetResult(dataset=ds) for ds in datasets}
            for algo in algorithms
        }
        
        # 每个算法的进度
        self.progress: Dict[str, AlgorithmProgress] = {
            algo: AlgorithmProgress(total=len(datasets))
            for algo in algorithms
        }
        
        # 算法停止状态
        self.stopped: Dict[str, bool] = {algo: False for algo in algorithms}
        
        # 最近的改进信息
        self.last_improvement: Optional[str] = None
        
        # 渲染相关
        self._last_print_lines: int = 0
    
    def mark_running(self, algorithm: str, dataset: str):
        """标记任务开始运行（线程安全）"""
        with self._lock:
            if algorithm not in self.results:
                return
            if dataset not in self.results[algorithm]:
                return
            
            self.results[algorithm][dataset].status = "running"
            
            # 更新算法状态
            if self.progress[algorithm].status == "pending":
                self.progress[algorithm].status = "running"
    
    def update(
        self,
        algorithm: str,
        dataset: str,
        metrics: Dict[str, float],
        hp: str = "-",
        mark_done: bool = True,
    ) -> Optional[str]:
        """
        更新结果（仅当新结果更优时）（线程安全）
        
        Returns:
            改进信息字符串（如果有改进）
        """
        with self._lock:
            if algorithm not in self.results:
                return None
            if dataset not in self.results[algorithm]:
                return None
            
            result = self.results[algorithm][dataset]
            old_value = result.metrics.get(self.primary_metric, 0)
            new_value = metrics.get(self.primary_metric, 0)
            
            improvement = None
            if new_value > old_value:
                if old_value > 0:
                    pct = (new_value - old_value) / old_value * 100
                    improvement = f"✓ {algorithm}/{dataset}: {self.primary_metric} {old_value:.4f} → {new_value:.4f} (+{pct:.1f}%) [{hp}]"
                else:
                    improvement = f"✓ {algorithm}/{dataset}: {self.primary_metric} = {new_value:.4f} [{hp}]"
                
                result.update(metrics, hp, mark_done=mark_done)
                self.last_improvement = improvement

            if mark_done:
                result.status = "done"
            return improvement
    
    def mark_completed(self, algorithm: str, dataset: str, success: bool = True):
        """标记任务完成（线程安全）"""
        with self._lock:
            self.completed_tasks += 1
            
            if algorithm not in self.progress:
                return
            
            prog = self.progress[algorithm]
            prog.completed += 1
            
            if success:
                prog.success += 1
            else:
                prog.failed += 1
            
            # 更新算法整体状态
            if prog.is_done:
                prog.status = "done" if prog.failed == 0 else "failed"
    
    def mark_algorithm_stopped(self, algorithm: str):
        """标记算法被停止（fail-fast 触发）（线程安全）"""
        with self._lock:
            self.stopped[algorithm] = True
            if algorithm in self.progress:
                self.progress[algorithm].status = "stopped"
    
    def _get_status_icon(self, algorithm: str, dataset: str) -> str:
        """获取状态图标"""
        if self.stopped.get(algorithm):
            return self.STATUS_ICONS["stopped"]
        
        result = self.results.get(algorithm, {}).get(dataset)
        if not result:
            return self.STATUS_ICONS["pending"]
        
        return self.STATUS_ICONS.get(result.status, self.STATUS_ICONS["pending"])
    
    def _format_value(self, value: Optional[float], status: str) -> str:
        """格式化指标值"""
        if value is None or value == 0:
            if status == "running":
                return "[running]"
            return "..."
        return f"{value:.4f}"
    
    def _progress_bar(self, width: int = 30) -> str:
        """生成总进度条"""
        if self.total_tasks == 0:
            return " " * width
        
        filled = int(width * self.completed_tasks / self.total_tasks)
        bar = "█" * filled + "░" * (width - filled)
        pct = self.completed_tasks / self.total_tasks * 100
        return f"[{bar}]  {self.completed_tasks}/{self.total_tasks}  ({pct:4.0f}%)"
    
    def render(self) -> str:
        """渲染表格为字符串（线程安全）"""
        with self._lock:
            return self._render_unsafe()
    
    def _render_unsafe(self) -> str:
        """渲染表格（不加锁，内部使用）"""
        lines = []
        
        # 计算列宽
        algo_width = max((len(a) for a in self.algorithms), default=10)
        algo_width = max(algo_width, 10)
        round_width = 5  # Round 列
        ds_width = max((len(ds) for ds in self.datasets), default=8)
        ds_width = max(ds_width, 8)
        metric_width = 10
        status_width = 8
        hp_width = 10
        
        # 表格总宽度
        num_metrics = min(len(self.metrics), 4)  # 最多显示 4 个指标
        total_width = (
            algo_width + 3 +
            round_width + 3 +  # Round 列
            ds_width + 3 +
            (metric_width + 3) * num_metrics +
            status_width + 3 +
            hp_width + 2
        )
        
        # 顶部边框
        lines.append("┌" + "─" * total_width + "┐")
        
        # 标题
        title = "Multi-Algorithm Benchmark Results"
        lines.append("│" + title.center(total_width) + "│")
        
        # 分隔线
        lines.append("├" + "─" * total_width + "┤")
        
        # 进度条
        progress = self._progress_bar(30)
        lines.append("│  Progress: " + progress.ljust(total_width - 12) + "│")
        
        # 分隔线
        lines.append("├" + "─" * total_width + "┤")
        
        # 表头
        header = f"│ {'Algorithm':<{algo_width}} │ {'Round':^{round_width}} │ {'Dataset':<{ds_width}} │"
        for m in self.metrics[:num_metrics]:
            header += f" {m:^{metric_width}} │"
        header += f" {'Status':^{status_width}} │ {'HP':^{hp_width}} │"
        lines.append(header)
        
        # 表头分隔线
        sep = "├" + "─" * (algo_width + 2) + "┼" + "─" * (round_width + 2) + "┼" + "─" * (ds_width + 2) + "┼"
        for _ in range(num_metrics):
            sep += "─" * (metric_width + 2) + "┼"
        sep += "─" * (status_width + 2) + "┼" + "─" * (hp_width + 2) + "┤"
        lines.append(sep)
        
        # 数据行
        for i, algo in enumerate(self.algorithms):
            current_round = self.progress[algo].current_round if algo in self.progress else 0
            
            for j, ds in enumerate(self.datasets):
                result = self.results[algo][ds]
                
                # 算法名和轮次只在该算法的第一行显示
                algo_display = algo if j == 0 else ""
                round_display = str(current_round) if j == 0 else ""
                
                row = f"│ {algo_display:<{algo_width}} │ {round_display:^{round_width}} │ {ds:<{ds_width}} │"
                
                for m in self.metrics[:num_metrics]:
                    val = result.metrics.get(m)
                    val_str = self._format_value(val, result.status)
                    row += f" {val_str:^{metric_width}} │"
                
                status = self._get_status_icon(algo, ds)
                hp_display = result.best_hp[:hp_width] if len(result.best_hp) > hp_width else result.best_hp
                row += f" {status:^{status_width}} │ {hp_display:^{hp_width}} │"
                lines.append(row)
            
            # 算法之间的分隔线（除了最后一个）
            if i < len(self.algorithms) - 1:
                sep = "├" + "─" * (algo_width + 2) + "┼" + "─" * (round_width + 2) + "┼" + "─" * (ds_width + 2) + "┼"
                for _ in range(num_metrics):
                    sep += "─" * (metric_width + 2) + "┼"
                sep += "─" * (status_width + 2) + "┼" + "─" * (hp_width + 2) + "┤"
                lines.append(sep)
        
        # 底部边框
        lines.append("└" + "─" * total_width + "┘")
        
        return "\n".join(lines)
    
    def display(self, clear: bool = False, consume_improvement: bool = False):
        """显示表格（线程安全）"""
        if not sys.stdout.isatty():
            clear = False
        if clear and self._last_print_lines:
            sys.stdout.write(f"\033[{self._last_print_lines}A\033[J")
        
        # render() 内部已经加锁
        rendered = self.render()
        print(rendered)
        
        improvement_printed = False
        with self._lock:
            if self.last_improvement:
                print(f"\033[92m{self.last_improvement}\033[0m")
                improvement_printed = True
                if consume_improvement:
                    self.last_improvement = None
        
        self._last_print_lines = rendered.count("\n") + 1 + (1 if improvement_printed else 0)

    def set_status(self, algorithm: str, dataset: str, status: str) -> None:
        """设置任务状态（线程安全）"""
        with self._lock:
            if algorithm in self.results and dataset in self.results[algorithm]:
                self.results[algorithm][dataset].status = status
    
    def update_algorithm_round(self, algorithm: str, round_num: int) -> None:
        """更新算法的当前轮次（线程安全）"""
        with self._lock:
            if algorithm in self.progress:
                self.progress[algorithm].current_round = round_num
    
    def reset_algorithm_for_new_round(self, algorithm: str) -> None:
        """
        为新轮次重置算法状态（线程安全）
        
        在算法进入新一轮迭代时调用，重置数据集状态和进度计数。
        """
        with self._lock:
            if algorithm not in self.results:
                return
            
            # 重置所有数据集状态
            for ds in self.datasets:
                if ds in self.results[algorithm]:
                    self.results[algorithm][ds].status = "pending"
                    self.results[algorithm][ds].metrics = {}
                    self.results[algorithm][ds].best_hp = "-"
            
            # 重置进度（保留 current_round）
            if algorithm in self.progress:
                current_round = self.progress[algorithm].current_round
                self.progress[algorithm] = AlgorithmProgress(
                    total=len(self.datasets),
                    current_round=current_round + 1,
                )
            
            # 重置停止状态
            self.stopped[algorithm] = False
    
    def set_algorithm_total_tasks(self, algorithm: str, total: int) -> None:
        """设置算法的总任务数（线程安全）"""
        with self._lock:
            if algorithm in self.progress:
                self.progress[algorithm].total = total
    
    def get_best_results(self) -> Dict[str, Dict[str, DatasetResult]]:
        """获取所有结果（按算法分组）"""
        return {
            algo: ds_results.copy()
            for algo, ds_results in self.results.items()
        }
    
    def get_algorithm_summary(self) -> Dict[str, AlgorithmProgress]:
        """获取每个算法的进度摘要"""
        return self.progress.copy()
    
    def summary_dict(self) -> Dict[str, Any]:
        """返回摘要字典（用于 JSON 序列化）（线程安全）"""
        with self._lock:
            return {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "algorithms": {
                    algo: {
                        "progress": {
                            "total": self.progress[algo].total,
                            "completed": self.progress[algo].completed,
                            "success": self.progress[algo].success,
                            "failed": self.progress[algo].failed,
                            "status": self.progress[algo].status,
                            "current_round": self.progress[algo].current_round,
                        },
                        "stopped": self.stopped[algo],
                        "datasets": {
                            ds: {
                                "metrics": result.metrics,
                                "best_hp": result.best_hp,
                                "status": result.status,
                            }
                            for ds, result in algo_results.items()
                        }
                    }
                    for algo, algo_results in self.results.items()
                }
            }


# ==============================================================================
# 便捷函数
# ==============================================================================

def create_recsys_table(
    datasets: List[str],
    total_tasks: int = 0,
    metrics: List[str] = None
) -> ResultTable:
    """创建推荐系统结果表格"""
    return ResultTable(
        title="Current Best Test Metrics",
        datasets=datasets,
        metrics=metrics or ["recall@20", "recall@10", "ndcg@20", "ndcg@10"],
        total_tasks=total_tasks,
        primary_metric="recall@20"
    )


# 测试
if __name__ == "__main__":
    # 创建表格
    table = ResultTable(
        title="Current Best Test Metrics",
        datasets=["baby", "sports", "clothing"],
        metrics=["recall@20", "recall@10", "ndcg@20", "ndcg@10"],
        total_tasks=48
    )
    
    # 模拟第一次更新
    table.set_running("baby")
    table.set_running("sports")
    table.set_running("clothing")
    table.print()
    
    print()
    
    # 模拟更新
    table.update("baby", {
        "recall@20": 0.0504,
        "recall@10": 0.0297,
        "ndcg@20": 0.0214,
        "ndcg@10": 0.0161
    }, hp="combo_3")
    table.complete_task()
    table.print()
    
    print()
    
    # 模拟改进
    improvement = table.update("baby", {
        "recall@20": 0.0677,
        "recall@10": 0.0406,
        "ndcg@20": 0.0291,
        "ndcg@10": 0.0220
    }, hp="combo_2")
    table.complete_task()
    table.complete_task()
    table.print()
    
    if improvement:
        print(f"\033[92m{improvement}\033[0m")
