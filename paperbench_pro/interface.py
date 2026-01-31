"""
PaperBenchPro 统一接口 v2

基于新架构设计的统一接口，提供 CLI 和 Python API
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .configs import (
    load_task_config, load_cluster_config,
    TaskConfig, ClusterConfig,
    AlgorithmConfig, BatchRunConfig,
)
from .scheduler import GPUResource, NodeGPUScheduler, get_expander
from .executor import SSHSingularityExecutor, build_bind_paths, get_ssh_manager
from .collector import (
    ResultTable, TaskResult, RunSummary,
    BatchResultTable, BatchRunSummary,
    parse_output, save_results, generate_report,
    save_batch_results, generate_batch_report,
)
from .collector.mmrec_collector import MMRecCollector
from .collector.recbole_collector import RecBoleCollector
from .logs import setup_logging, get_logger, create_run_logger


@dataclass
class RunConfig:
    """
    运行配置
    
    整合所有运行参数
    """
    # 任务标识
    domain: str
    task: str
    model: str
    
    # 输入文件
    algorithm_path: Path
    hyperparameter_path: Path
    
    # 数据集（None 表示使用配置中的默认数据集）
    datasets: Optional[List[str]] = None
    
    # 调度选项
    expand_hyperparameters: bool = True      # 是否展开超参数
    max_concurrent: int = 16                 # 最大并发
    timeout: int = 10800                      # 单任务超时
    fail_fast: bool = True                   # 失败后是否立即停止（默认开启）
    
    # 输出选项
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_logs: bool = True
    quiet: bool = True                       # 静默模式（只显示关键更新）
    
    # 覆盖配置
    nodes: Optional[List[str]] = None        # 覆盖节点列表
    gpus_per_node: Optional[int] = None      # 覆盖每节点 GPU 数
    
    def __post_init__(self):
        self.algorithm_path = Path(self.algorithm_path)
        self.hyperparameter_path = Path(self.hyperparameter_path)
        self.output_dir = Path(self.output_dir)


class RenderScheduler:
    """Rate-limit table rendering to avoid interleaved output."""

    def __init__(self, render_fn, min_interval: float = 0.4) -> None:
        self.render_fn = render_fn
        self.min_interval = min_interval
        self.last_render = 0.0
        self.pending = False
        self.pending_clear = False
        self.rendered = False

    def request(self, clear: bool, force: bool = False) -> None:
        now = time.monotonic()
        if not self.rendered:
            self.render_fn(clear=False)
            self.rendered = True
            self.last_render = now
            self.pending = False
            self.pending_clear = False
            return
        if force or (now - self.last_render) >= self.min_interval:
            self.render_fn(clear=clear)
            self.last_render = now
            self.pending = False
            self.pending_clear = False
            return
        self.pending = True
        self.pending_clear = self.pending_clear or clear

    def flush(self) -> None:
        if not self.rendered:
            self.request(clear=False, force=True)
            return
        if self.pending:
            self.render_fn(clear=self.pending_clear)
            self.last_render = time.monotonic()
            self.pending = False
            self.pending_clear = False


class PaperBenchProV2:
    """
    PaperBenchPro 主类 v2
    
    协调配置、调度、执行、收集各模块
    
    Usage:
        bench = PaperBenchProV2()
        result = bench.run(RunConfig(
            domain="Recsys",
            task="MultiModal",
            model="BM3",
            algorithm_path=Path("workspace/algorithm.py"),
            hyperparameter_path=Path("workspace/hyperparameter.yaml"),
        ))
    """
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path(__file__).parent / "configs"
        self.logger = get_logger("interface")
    
    def run(self, run_config: RunConfig) -> Dict[str, Any]:
        """
        执行完整的 benchmark 流程（同步版本）
        
        Args:
            run_config: 运行配置
            
        Returns:
            Dict: 运行结果
        """
        # 1. 加载配置
        self.logger.info(f"Loading config for {run_config.domain}/{run_config.task}")
        task_config = load_task_config(run_config.domain, run_config.task, self.config_dir)
        cluster_config = load_cluster_config(self.config_dir)
        
        # 2. 初始化日志
        run_logger = None
        if run_config.save_logs:
            run_logger = create_run_logger(run_config.model)
            self.logger.info(f"Logs will be saved to: {run_logger.run_dir}")
        
        # 3. 确定数据集
        datasets = run_config.datasets
        if not datasets and task_config.data:
            datasets = task_config.data.datasets
        if not datasets:
            datasets = ["default"]
        self.logger.info(f"Datasets: {datasets}")
        
        # 4. 展开超参数（如果需要）
        hp_files = [run_config.hyperparameter_path]
        if run_config.expand_hyperparameters:
            try:
                expander = get_expander(run_config.domain, run_config.task)
                hp_output_dir = run_config.output_dir / "hyperparameters" / run_config.model
                hp_output_dir.mkdir(parents=True, exist_ok=True)
                hp_files = expander.expand(run_config.hyperparameter_path, hp_output_dir)
                self.logger.info(f"Expanded to {len(hp_files)} hyperparameter configs")
            except ValueError:
                self.logger.warning("No expander registered, using original hyperparameter file")
        
        # 5. 生成任务列表
        tasks = []
        for dataset in datasets:
            for hp_path in hp_files:
                tasks.append({
                    "dataset": dataset,
                    "hp_path": hp_path,
                    "hp_name": hp_path.stem,
                })
        
        self.logger.info(f"Generated {len(tasks)} tasks ({len(datasets)} datasets × {len(hp_files)} hp)")
        
        # 6. 创建结果摘要
        summary = RunSummary(
            domain=run_config.domain,
            task=run_config.task,
            model=run_config.model,
            total_tasks=len(tasks),
        )
        
        # 7. 创建实时追踪表格（从配置加载）
        if not task_config.eval:
            raise ValueError(f"Task config missing 'eval' section: {run_config.domain}/{run_config.task}")
        
        primary_metric = task_config.eval.primary_metric
        metrics_list = task_config.eval.metrics[:4]  # 最多显示4个
        
        tracker = ResultTable(
            title="Current Best Test Metrics",
            datasets=datasets,
            metrics=metrics_list,
            total_tasks=len(tasks),
            primary_metric=primary_metric,
        )

        import threading
        tracker_lock = threading.Lock()

        def _render(clear: bool) -> None:
            tracker.print(clear=clear, consume_improvement=True)

        renderer = RenderScheduler(_render)

        def render_tracker(clear: bool, force: bool = False) -> None:
            renderer.request(clear=clear, force=force)
        tasks_per_dataset = {ds: 0 for ds in datasets}
        for task in tasks:
            tasks_per_dataset[task["dataset"]] += 1
        completed_by_dataset = {ds: 0 for ds in datasets}
        use_mmrec_stream = (task_config.framework or "").lower() == "mmrec"
        
        # 8. 设置 GPU 调度
        nodes = run_config.nodes or (cluster_config.ssh.nodes if cluster_config.ssh else ["localhost"])
        gpus_per_node = run_config.gpus_per_node or (cluster_config.scheduler.gpus_per_node if cluster_config.scheduler else 8)
        max_per_gpu = cluster_config.scheduler.max_per_gpu if cluster_config.scheduler else 2
        
        # 创建 GPU 资源列表
        resources = [
            GPUResource(node=node, gpu_id=gpu_id)
            for node in nodes
            for gpu_id in range(gpus_per_node)
        ]
        
        scheduler = NodeGPUScheduler(resources, max_per_gpu=max_per_gpu)
        self.logger.info(f"GPU pool: {len(nodes)} nodes × {gpus_per_node} GPUs")
        
        # 8.5 预建立 SSH 连接（避免并发连接过多导致被拒绝）
        if nodes and nodes != ["localhost"]:
            ssh_manager = get_ssh_manager()
            self.logger.info(f"Establishing SSH connections to {len(nodes)} nodes...")
            conn_results = ssh_manager.ensure_connections(nodes)
            failed_nodes = [n for n, ok in conn_results.items() if not ok]
            if failed_nodes:
                self.logger.warning(f"Failed to establish SSH to: {failed_nodes}")
        
        # 9. 执行任务
        def run_single_task(task_info: dict, resource: Optional[GPUResource], stop_event: threading.Event) -> TaskResult:
            dataset = task_info["dataset"]
            hp_path = task_info["hp_path"]
            hp_name = task_info["hp_name"]
            
            node = resource.node if resource else "localhost"
            gpu_id = resource.gpu_id if resource else 0

            stream_collector = None
            if run_config.domain == "Recsys":
                stream_collector = MMRecCollector() if use_mmrec_stream else RecBoleCollector()

            # 标记数据集开始运行
            with tracker_lock:
                prev_status = tracker.results.get(dataset).status if dataset in tracker.results else None
                tracker.mark_running(dataset)
                if prev_status != "running":
                    render_tracker(clear=True)
            
            # 记录任务启动（只使用 run_logger，避免重复日志）
            if run_logger:
                run_logger.log_task_start(dataset, hp_name, node, gpu_id)
            else:
                self.logger.info(f"Task started: {dataset}/{hp_name} on {node}:GPU{gpu_id}")
            
            import time
            start_time = time.time()
            
            def make_line_callback(log_callback=None):
                def _callback(line: str, is_stderr: bool) -> None:
                    if log_callback:
                        log_callback(line, is_stderr)
                    if is_stderr or stream_collector is None:
                        return
                    metrics = stream_collector.feed(line)
                    if metrics:
                        with tracker_lock:
                            improvement = tracker.update(
                                dataset,
                                metrics,
                                hp_name,
                                mark_done=False,
                            )
                            if improvement:
                                render_tracker(clear=True)
                return _callback

            try:
                # 构建挂载路径
                bind_paths = build_bind_paths(
                    task_config=task_config,
                    shared_root=cluster_config.shared_root,
                    model=run_config.model,
                    dataset=dataset,
                    algorithm_path=run_config.algorithm_path,
                    hyperparameter_path=hp_path,
                    log_dir=run_logger.tasks_dir if run_logger else None,
                )
                
                # 创建执行器
                sif_path = str(Path(cluster_config.shared_root) / "paperbench_pro" / task_config.execution.container)
                
                executor = SSHSingularityExecutor(
                    sif_path=sif_path,
                    bind_paths=bind_paths,
                    node=node,
                    gpu_id=gpu_id,
                    singularity_exe=cluster_config.singularity.executable if cluster_config.singularity else "singularity",
                )
                
                # 构建命令（从配置模板）
                workdir = task_config.execution.workdir if task_config.execution else "/app"
                
                # 从 args.template 构建命令
                if task_config.args and "template" in task_config.args:
                    # 替换模板变量
                    args_str = task_config.args["template"].format(
                        model=run_config.model,
                        dataset=dataset,
                        gpu=gpu_id,
                    )
                    cmd = ["python", "replicate.py"] + args_str.split()
                else:
                    # 使用 execution.entry（如 "python run_recbole.py -m {model} -d {dataset}"）
                    entry_str = task_config.execution.entry.format(
                        model=run_config.model,
                        dataset=dataset,
                        gpu=gpu_id,
                    )
                    cmd = entry_str.split()
                
                # 执行（传递 stop_event 支持 fast fail，使用流式日志）
                if run_logger:
                    # 使用流式日志：实时写入每行输出到日志文件
                    with run_logger.stream_task_output(dataset, hp_name) as log_callback:
                        line_callback = make_line_callback(log_callback)
                        result = executor.execute(
                            command=cmd,
                            work_dir=workdir,
                            timeout=run_config.timeout,
                            streaming=False,
                            line_callback=line_callback,
                            stop_event=stop_event,
                        )
                else:
                    line_callback = make_line_callback()
                    result = executor.execute(
                        command=cmd,
                        work_dir=workdir,
                        timeout=run_config.timeout,
                        streaming=False,
                        line_callback=line_callback,
                        stop_event=stop_event,
                    )
                
                # 解析结果
                metrics = parse_output(
                    stdout=result.stdout,
                    domain=run_config.domain,
                    task=run_config.task,
                )
                
                # 创建任务结果
                success = result.return_code == 0 and len(metrics) > 0
                
                # 构建错误信息
                error_msg = None
                if not success:
                    if result.return_code != 0:
                        error_msg = f"Exit code {result.return_code}"
                    elif len(metrics) == 0:
                        error_msg = "No metrics found in output"
                    else:
                        error_msg = "Unknown error"
                
                parsed = TaskResult(
                    dataset=dataset,
                    hp_name=hp_name,
                    metrics=metrics,
                    success=success,
                    duration=time.time() - start_time,
                    error=error_msg,
                    raw_stdout=result.stdout if not success else None,  # 失败时保存完整输出
                    raw_stderr=result.stderr if not success else None,
                    node=node,
                    gpu_id=gpu_id,
                )
                
                # 标记完成并更新状态
                with tracker_lock:
                    tracker.mark_completed(dataset, parsed.success)
                    completed_by_dataset[dataset] += 1

                    # 更新最佳指标（如果成功）
                    if parsed.success and parsed.metrics:
                        improvement = tracker.update(
                            dataset,
                            parsed.metrics,
                            hp_name,
                            mark_done=False,
                        )
                        if improvement:
                            render_tracker(clear=True)

                    # 数据集全部任务完成后标记状态
                    if completed_by_dataset[dataset] >= tasks_per_dataset[dataset]:
                        status = "done" if tracker.results[dataset].fail_count == 0 else "failed"
                        tracker.set_status(dataset, status)
                        render_tracker(clear=True)
                
                # 记录任务结束
                if run_logger:
                    run_logger.log_task_end(dataset, hp_name, parsed.success, parsed.duration)
                else:
                    status = "SUCCESS" if parsed.success else "FAILED"
                    self.logger.info(f"Task {status}: {dataset}/{hp_name} ({parsed.duration:.1f}s)")
                
                return parsed
                
            except Exception as e:
                self.logger.error(f"Task failed: {dataset}/{hp_name} - {e}")
                with tracker_lock:
                    tracker.mark_completed(dataset, False)
                    completed_by_dataset[dataset] += 1
                    if completed_by_dataset[dataset] >= tasks_per_dataset[dataset]:
                        status = "done" if tracker.results[dataset].fail_count == 0 else "failed"
                        tracker.set_status(dataset, status)
                        render_tracker(clear=True)
                
                parsed = TaskResult(
                    dataset=dataset,
                    hp_name=hp_name,
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e),
                    node=node,
                    gpu_id=gpu_id,
                )
                
                if run_logger:
                    run_logger.log_task_end(dataset, hp_name, False, parsed.duration)
                
                return parsed
        
        # 定义快速失败判断函数
        def should_stop_on_error(result: TaskResult) -> bool:
            """判断是否应该停止所有任务"""
            if result is None:
                return False
            if result.success:
                return False
            # 任何失败都触发 fast fail（根据用户需求）
            return True
        
        # 运行所有任务（支持快速失败）
        # max_workers 需要考虑 max_per_gpu，允许每个GPU同时运行多个任务
        max_workers = min(run_config.max_concurrent, len(resources) * max_per_gpu)
        results, fatal_result = scheduler.run_batch(
            tasks=tasks,
            run_fn=run_single_task,
            max_workers=max_workers,
            fail_fast=run_config.fail_fast,
            should_stop=should_stop_on_error,
        )
        
        # 检查是否快速失败
        if fatal_result is not None:
            with tracker_lock:
                render_tracker(clear=True, force=True)
            self.logger.error(f"Fatal error detected, stopping all tasks")
            self.logger.error(f"Error: {fatal_result.error}")
            print("\n" + "=" * 70)
            print("\033[31m  FATAL ERROR - All tasks stopped\033[0m")
            print("=" * 70)
            print(f"  Dataset: {fatal_result.dataset}")
            print(f"  Error: {fatal_result.error}")
            # 显示完整输出的最后部分（用于调试）
            if fatal_result.raw_stdout:
                last_lines = fatal_result.raw_stdout.strip().split('\n')[-30:]
                print("\n  Last 30 lines of stdout:")
                for line in last_lines:
                    print(f"    {line}")
            if fatal_result.raw_stderr:
                print("\n  Stderr:")
                for line in fatal_result.raw_stderr.strip().split('\n')[-10:]:
                    print(f"    {line}")
            print("=" * 70 + "\n")
        
        # 10. 聚合结果
        for task_info, result in results:
            if result is not None:
                summary.add_result(result)
        
        # 11. 完成摘要并保存
        summary.finalize(tracker.get_best_results())
        result_path = save_results(summary, run_config.output_dir / "results")
        self.logger.info(f"Results saved to: {result_path}")
        
        if run_logger:
            run_logger.log_summary(
                summary.total_tasks,
                summary.success,
                summary.failed,
                0,  # duration 在 summary 里计算
            )
        
        # 12. 显示实时追踪汇总
        with tracker_lock:
            render_tracker(clear=True, force=True)
        
        # 13. 输出报告
        report = generate_report(summary)
        print(report)
        
        # 14. 返回结果（包含实时追踪信息）
        result_dict = summary.to_dict()
        result_dict["realtime_tracking"] = tracker.summary_dict()
        
        # 添加 overall best 信息
        overall_best = tracker.get_overall_best()
        if overall_best:
            result_dict["best"] = {
                "dataset": overall_best.dataset,
                "metrics": overall_best.metrics,
                "hp_info": overall_best.hp_info,
            }
        
        # 添加致命错误信息（供 AI Scientist 使用）
        if fatal_result is not None:
            result_dict["fatal_error"] = {
                "stdout": fatal_result.raw_stdout,
                "stderr": fatal_result.raw_stderr,
            }
        else:
            result_dict["fatal_error"] = None
        
        return result_dict

    def run_batch(self, batch_config: BatchRunConfig) -> Dict[str, Any]:
        """
        批量执行多个算法（支持按算法粒度的 fail-fast）
        
        当某个算法失败时，只停止该算法的后续任务，其他算法继续执行。
        所有算法共享同一个 GPU 资源池，实现最大化并行。
        
        Args:
            batch_config: 批量运行配置
            
        Returns:
            Dict: 批量运行结果，按算法分组
            
        Example:
            bench = PaperBenchProV2()
            result = bench.run_batch(BatchRunConfig(
                domain="Recsys",
                task="MultiModal",
                algorithms=[
                    AlgorithmConfig("BM3", "path/to/bm3.py", "path/to/bm3_hp.yaml"),
                    AlgorithmConfig("LATTICE", "path/to/lattice.py", "path/to/lattice_hp.yaml"),
                ],
                datasets=["baby", "sports"],
            ))
        """
        import threading
        import time
        
        # 1. 加载配置
        self.logger.info(f"Loading config for {batch_config.domain}/{batch_config.task}")
        task_config = load_task_config(batch_config.domain, batch_config.task, self.config_dir)
        cluster_config = load_cluster_config(self.config_dir)
        
        # 2. 初始化日志
        run_logger = None
        if batch_config.save_logs:
            algo_names = "_".join(a.name for a in batch_config.algorithms[:3])
            run_logger = create_run_logger(f"batch_{algo_names}")
            self.logger.info(f"Logs will be saved to: {run_logger.run_dir}")
        
        # 3. 确定数据集
        datasets = batch_config.datasets
        if not datasets and task_config.data:
            datasets = task_config.data.datasets
        if not datasets:
            datasets = ["default"]
        self.logger.info(f"Datasets: {datasets}")
        
        # 4. 为每个算法展开超参数，生成扁平化任务列表
        tasks = []
        algo_hp_counts: Dict[str, int] = {}
        
        for algo in batch_config.algorithms:
            hp_files = [algo.hyperparameter_path]
            
            if batch_config.expand_hyperparameters:
                try:
                    expander = get_expander(batch_config.domain, batch_config.task)
                    hp_output_dir = batch_config.output_dir / "hyperparameters" / algo.name
                    hp_output_dir.mkdir(parents=True, exist_ok=True)
                    hp_files = expander.expand(algo.hyperparameter_path, hp_output_dir)
                    self.logger.info(f"[{algo.name}] Expanded to {len(hp_files)} hyperparameter configs")
                except ValueError:
                    self.logger.warning(f"[{algo.name}] No expander registered, using original")
            
            algo_hp_counts[algo.name] = len(hp_files)
            
            for dataset in datasets:
                for hp_path in hp_files:
                    tasks.append({
                        "algorithm": algo.name,
                        "algorithm_path": algo.algorithm_path,
                        "dataset": dataset,
                        "hp_path": hp_path,
                        "hp_name": hp_path.stem,
                    })
        
        total_tasks = len(tasks)
        self.logger.info(
            f"Generated {total_tasks} tasks "
            f"({len(batch_config.algorithms)} algorithms × {len(datasets)} datasets)"
        )
        
        # 5. 验证配置
        if not task_config.eval:
            raise ValueError(f"Task config missing 'eval' section: {batch_config.domain}/{batch_config.task}")
        
        primary_metric = task_config.eval.primary_metric
        metrics_list = task_config.eval.metrics[:4]
        
        # 6. 创建摘要和追踪表格
        summary = BatchRunSummary(
            domain=batch_config.domain,
            task=batch_config.task,
            algorithms=batch_config.algorithm_names,
            total_tasks=total_tasks,
        )
        
        tracker = BatchResultTable(
            algorithms=batch_config.algorithm_names,
            datasets=datasets,
            metrics=metrics_list,
            total_tasks=total_tasks,
            primary_metric=primary_metric,
        )

        tracker_lock = threading.Lock()

        def _render(clear: bool) -> None:
            tracker.display(clear=clear, consume_improvement=True)

        renderer = RenderScheduler(_render)

        def render_tracker(clear: bool, force: bool = False) -> None:
            renderer.request(clear=clear, force=force)
        tasks_per_algo_dataset: Dict[tuple, int] = {}
        for task in tasks:
            key = (task["algorithm"], task["dataset"])
            tasks_per_algo_dataset[key] = tasks_per_algo_dataset.get(key, 0) + 1
        completed_by_algo_dataset = {key: 0 for key in tasks_per_algo_dataset}
        use_mmrec_stream = (task_config.framework or "").lower() == "mmrec"
        
        # 7. 设置 GPU 调度
        nodes = batch_config.nodes or (cluster_config.ssh.nodes if cluster_config.ssh else ["localhost"])
        gpus_per_node = batch_config.gpus_per_node or (cluster_config.scheduler.gpus_per_node if cluster_config.scheduler else 8)
        max_per_gpu = cluster_config.scheduler.max_per_gpu if cluster_config.scheduler else 2
        
        resources = [
            GPUResource(node=node, gpu_id=gpu_id)
            for node in nodes
            for gpu_id in range(gpus_per_node)
        ]
        
        scheduler = NodeGPUScheduler(resources, max_per_gpu=max_per_gpu)
        self.logger.info(f"GPU pool: {len(nodes)} nodes × {gpus_per_node} GPUs")
        
        # 7.5 预建立 SSH 连接（避免并发连接过多导致被拒绝）
        if nodes and nodes != ["localhost"]:
            ssh_manager = get_ssh_manager()
            self.logger.info(f"Establishing SSH connections to {len(nodes)} nodes...")
            conn_results = ssh_manager.ensure_connections(nodes)
            failed_nodes = [n for n, ok in conn_results.items() if not ok]
            if failed_nodes:
                self.logger.warning(f"Failed to establish SSH to: {failed_nodes}")
        
        # 8. 定义任务执行函数
        def run_single_task(
            task_info: dict,
            resource: Optional[GPUResource],
            group_stop_event: threading.Event,
        ) -> TaskResult:
            algorithm = task_info["algorithm"]
            algorithm_path = task_info["algorithm_path"]
            dataset = task_info["dataset"]
            hp_path = task_info["hp_path"]
            hp_name = task_info["hp_name"]
            
            node = resource.node if resource else "localhost"
            gpu_id = resource.gpu_id if resource else 0

            stream_collector = MMRecCollector() if use_mmrec_stream else None

            # 标记运行
            with tracker_lock:
                result = tracker.results.get(algorithm, {}).get(dataset)
                prev_status = result.status if result else None
                tracker.mark_running(algorithm, dataset)
                if prev_status != "running":
                    render_tracker(clear=True)
            
            if run_logger:
                run_logger.log_task_start(f"{algorithm}/{dataset}", hp_name, node, gpu_id)
            
            start_time = time.time()
            
            def make_line_callback(log_callback=None):
                def _callback(line: str, is_stderr: bool) -> None:
                    if log_callback:
                        log_callback(line, is_stderr)
                    if is_stderr or stream_collector is None:
                        return
                    metrics = stream_collector.feed(line)
                    if metrics:
                        with tracker_lock:
                            improvement = tracker.update(
                                algorithm,
                                dataset,
                                metrics,
                                hp_name,
                                mark_done=False,
                            )
                            if improvement:
                                render_tracker(clear=True)
                return _callback

            try:
                # 构建挂载路径
                bind_paths = build_bind_paths(
                    task_config=task_config,
                    shared_root=cluster_config.shared_root,
                    model=algorithm,
                    dataset=dataset,
                    algorithm_path=algorithm_path,
                    hyperparameter_path=hp_path,
                    log_dir=run_logger.tasks_dir if run_logger else None,
                )
                
                # 创建执行器
                sif_path = str(Path(cluster_config.shared_root) / "paperbench_pro" / task_config.execution.container)
                
                executor = SSHSingularityExecutor(
                    sif_path=sif_path,
                    bind_paths=bind_paths,
                    node=node,
                    gpu_id=gpu_id,
                    singularity_exe=cluster_config.singularity.executable if cluster_config.singularity else "singularity",
                )
                
                # 构建命令
                workdir = task_config.execution.workdir if task_config.execution else "/app"
                
                if task_config.args and "template" in task_config.args:
                    args_str = task_config.args["template"].format(
                        model=algorithm,
                        dataset=dataset,
                        gpu=gpu_id,
                    )
                    cmd = ["python", "replicate.py"] + args_str.split()
                else:
                    entry_str = task_config.execution.entry.format(
                        model=algorithm,
                        dataset=dataset,
                        gpu=gpu_id,
                    )
                    cmd = entry_str.split()
                
                # 执行（使用流式日志）
                if run_logger:
                    # 使用流式日志：实时写入每行输出到日志文件
                    # 对于 batch 模式，日志文件名包含算法名
                    with run_logger.stream_task_output(f"{algorithm}_{dataset}", hp_name) as log_callback:
                        line_callback = make_line_callback(log_callback)
                        result = executor.execute(
                            command=cmd,
                            work_dir=workdir,
                            timeout=batch_config.timeout,
                            streaming=False,
                            line_callback=line_callback,
                            stop_event=group_stop_event,
                        )
                else:
                    line_callback = make_line_callback()
                    result = executor.execute(
                        command=cmd,
                        work_dir=workdir,
                        timeout=batch_config.timeout,
                        streaming=False,
                        line_callback=line_callback,
                        stop_event=group_stop_event,
                    )
                
                # 解析结果
                metrics = parse_output(
                    stdout=result.stdout,
                    domain=batch_config.domain,
                    task=batch_config.task,
                )
                
                success = result.return_code == 0 and len(metrics) > 0
                
                error_msg = None
                if not success:
                    if result.return_code != 0:
                        error_msg = f"Exit code {result.return_code}"
                    elif len(metrics) == 0:
                        error_msg = "No metrics found in output"
                
                parsed = TaskResult(
                    algorithm=algorithm,
                    dataset=dataset,
                    hp_name=hp_name,
                    metrics=metrics,
                    success=success,
                    duration=time.time() - start_time,
                    error=error_msg,
                    raw_stdout=result.stdout if not success else None,
                    raw_stderr=result.stderr if not success else None,
                    node=node,
                    gpu_id=gpu_id,
                )
                
                # 更新追踪表格
                with tracker_lock:
                    tracker.mark_completed(algorithm, dataset, parsed.success)
                    key = (algorithm, dataset)
                    if key in completed_by_algo_dataset:
                        completed_by_algo_dataset[key] += 1

                    if parsed.success and parsed.metrics:
                        improvement = tracker.update(
                            algorithm,
                            dataset,
                            parsed.metrics,
                            hp_name,
                            mark_done=False,
                        )
                        if improvement:
                            render_tracker(clear=True)

                    if key in tasks_per_algo_dataset and completed_by_algo_dataset.get(key, 0) >= tasks_per_algo_dataset[key]:
                        status = "done" if tracker.results[algorithm][dataset].fail_count == 0 else "failed"
                        tracker.set_status(algorithm, dataset, status)
                        render_tracker(clear=True)
                
                if run_logger:
                    run_logger.log_task_end(f"{algorithm}/{dataset}", hp_name, parsed.success, parsed.duration)
                
                return parsed
                
            except Exception as e:
                self.logger.error(f"Task failed: {algorithm}/{dataset}/{hp_name} - {e}")
                with tracker_lock:
                    tracker.mark_completed(algorithm, dataset, False)
                    key = (algorithm, dataset)
                    if key in completed_by_algo_dataset:
                        completed_by_algo_dataset[key] += 1
                    if key in tasks_per_algo_dataset and completed_by_algo_dataset.get(key, 0) >= tasks_per_algo_dataset[key]:
                        status = "done" if tracker.results[algorithm][dataset].fail_count == 0 else "failed"
                        tracker.set_status(algorithm, dataset, status)
                        render_tracker(clear=True)
                
                return TaskResult(
                    algorithm=algorithm,
                    dataset=dataset,
                    hp_name=hp_name,
                    success=False,
                    duration=time.time() - start_time,
                    error=str(e),
                    node=node,
                    gpu_id=gpu_id,
                )
        
        # 9. 定义 fail-fast 判断
        def should_stop_on_error(result: TaskResult) -> bool:
            if result is None:
                return False
            return not result.success
        
        # 10. 运行所有任务（分组 fail-fast）
        max_workers = min(batch_config.max_concurrent, len(resources) * max_per_gpu)
        
        results, fatal_by_group = scheduler.run_batch_grouped(
            tasks=tasks,
            run_fn=run_single_task,
            group_key=lambda t: t["algorithm"],
            max_workers=max_workers,
            fail_fast_per_group=batch_config.fail_fast_per_algorithm,
            should_stop=should_stop_on_error,
        )
        
        # 11. 处理结果
        for task_info, result in results:
            if result is not None:
                summary.add_result(result)
        
        # 标记停止的算法并记录致命错误
        for algo, fatal_result in fatal_by_group.items():
            tracker.mark_algorithm_stopped(algo)
            summary.set_fatal(algo, fatal_result)
            
            self.logger.error(f"[{algo}] Fatal error, algorithm stopped")
            print(f"\n\033[31m[{algo}] FATAL ERROR - Algorithm stopped\033[0m")
            print(f"  Dataset: {fatal_result.dataset}")
            print(f"  Error: {fatal_result.error}")
            if fatal_result.raw_stdout:
                last_lines = fatal_result.raw_stdout.strip().split('\n')[-15:]
                print("  Last 15 lines of stdout:")
                for line in last_lines:
                    print(f"    {line}")
        
        # 12. 完成摘要
        summary.finalize(tracker.get_best_results())
        
        # 保存结果
        result_path = save_batch_results(summary, batch_config.output_dir / "results")
        self.logger.info(f"Results saved to: {result_path}")
        
        if run_logger:
            success_count = sum(1 for _, r in results if r and r.success)
            failed_count = sum(1 for _, r in results if r and not r.success)
            run_logger.log_summary(total_tasks, success_count, failed_count, 0)
        
        # 13. 显示结果
        with tracker_lock:
            render_tracker(clear=True, force=True)
        report = generate_batch_report(summary)
        print(report)
        
        # 14. 返回结果
        result_dict = summary.to_dict()
        result_dict["realtime_tracking"] = tracker.summary_dict()
        
        return result_dict


# ==============================================================================
# 便捷函数
# ==============================================================================

def run(
    domain: str,
    task: str,
    model: str,
    algorithm_path: str,
    hyperparameter_path: str,
    datasets: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    便捷运行函数
    
    Args:
        domain: 领域 (Recsys, TimeSeries, etc.)
        task: 任务 (MultiModal, LongTermForecast, etc.)
        model: 模型名
        algorithm_path: 算法文件路径
        hyperparameter_path: 超参数文件路径
        datasets: 数据集列表
        **kwargs: 其他配置
        
    Returns:
        Dict: 运行结果
    """
    # 创建配置
    config = RunConfig(
        domain=domain,
        task=task,
        model=model,
        algorithm_path=Path(algorithm_path),
        hyperparameter_path=Path(hyperparameter_path),
        datasets=datasets,
        **kwargs,
    )
    
    # 初始化日志
    setup_logging(console=not config.quiet)
    
    # 运行
    bench = PaperBenchProV2()
    return bench.run(config)


def run_batch(
    domain: str,
    task: str,
    algorithms: List[Dict],
    datasets: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    批量运行便捷函数
    
    同时运行多个算法，支持按算法粒度的 fail-fast。
    当某个算法失败时，只停止该算法的后续任务，其他算法继续执行。
    
    Args:
        domain: 领域 (Recsys, TimeSeries, etc.)
        task: 任务 (MultiModal, LongTermForecast, etc.)
        algorithms: 算法列表，每个元素为 dict:
            {
                "name": "BM3",
                "algorithm_path": "path/to/algorithm.py",
                "hyperparameter_path": "path/to/hp.yaml",
            }
        datasets: 数据集列表（None 表示使用配置中的默认数据集）
        **kwargs: 其他配置选项:
            - expand_hyperparameters: bool = True
            - max_concurrent: int = 16
            - timeout: int = 3600
            - fail_fast_per_algorithm: bool = True
            - output_dir: Path = Path("output")
            - save_logs: bool = True
            - nodes: Optional[List[str]] = None
            - gpus_per_node: Optional[int] = None
        
    Returns:
        Dict: 批量运行结果，包含:
            - algorithms: 算法列表
            - results_by_algorithm: 按算法分组的详细结果
            - best_by_algorithm: 每个算法的最佳结果
            - fatal_by_algorithm: 每个算法的致命错误（如果有）
        
    Example:
        result = run_batch(
            domain="Recsys",
            task="MultiModal",
            algorithms=[
                {
                    "name": "BM3",
                    "algorithm_path": "workspace/bm3/algorithm.py",
                    "hyperparameter_path": "workspace/bm3/hp.yaml",
                },
                {
                    "name": "LATTICE",
                    "algorithm_path": "workspace/lattice/algorithm.py",
                    "hyperparameter_path": "workspace/lattice/hp.yaml",
                },
            ],
            datasets=["baby", "sports", "clothing"],
        )
        
        # 检查结果
        for algo, best in result["best_by_algorithm"].items():
            print(f"{algo}: {best}")
    """
    config = BatchRunConfig(
        domain=domain,
        task=task,
        algorithms=[AlgorithmConfig.from_dict(a) for a in algorithms],
        datasets=datasets,
        **kwargs,
    )
    
    setup_logging(console=not config.quiet)
    
    bench = PaperBenchProV2()
    return bench.run_batch(config)


# ==============================================================================
# 共享资源版本（支持多算法独立迭代）
# ==============================================================================

@dataclass
class SharedResources:
    """
    共享资源容器
    
    用于多算法并行独立迭代场景，各算法共享：
    - GPU 调度器
    - 结果表格
    - 配置信息
    """
    # GPU 调度
    scheduler: NodeGPUScheduler
    
    # 结果表格
    result_table: BatchResultTable
    
    # 配置
    task_config: TaskConfig
    cluster_config: ClusterConfig
    
    # 渲染控制
    renderer: RenderScheduler
    
    # 数据集列表
    datasets: List[str] = field(default_factory=list)
    
    # 指标列表
    metrics: List[str] = field(default_factory=list)
    
    # 主指标
    primary_metric: str = "recall@20"


def create_shared_resources(
    domain: str,
    task: str,
    algorithms: List[str],
    datasets: List[str] = None,
    config_dir: Path = None,
    nodes: List[str] = None,
    gpus_per_node: int = None,
) -> SharedResources:
    """
    创建共享资源（用于多算法独立迭代）
    
    Args:
        domain: 领域
        task: 任务
        algorithms: 算法名列表
        datasets: 数据集列表（None 表示使用配置中的默认数据集）
        config_dir: 配置目录
        nodes: 节点列表（覆盖配置）
        gpus_per_node: 每节点 GPU 数（覆盖配置）
    
    Returns:
        SharedResources 实例
    """
    config_dir = config_dir or Path(__file__).parent / "configs"
    
    # 加载配置
    task_config = load_task_config(domain, task, config_dir)
    cluster_config = load_cluster_config(config_dir)
    
    # 确定数据集
    if not datasets and task_config.data:
        datasets = task_config.data.datasets
    if not datasets:
        datasets = ["default"]
    
    # 确定指标
    if not task_config.eval:
        raise ValueError(f"Task config missing 'eval' section: {domain}/{task}")
    
    primary_metric = task_config.eval.primary_metric
    metrics_list = task_config.eval.metrics[:4]
    
    # 创建 GPU 调度器
    resolved_nodes = nodes or (cluster_config.ssh.nodes if cluster_config.ssh else ["localhost"])
    resolved_gpus = gpus_per_node or (cluster_config.scheduler.gpus_per_node if cluster_config.scheduler else 8)
    max_per_gpu = cluster_config.scheduler.max_per_gpu if cluster_config.scheduler else 2
    
    resources = [
        GPUResource(node=node, gpu_id=gpu_id)
        for node in resolved_nodes
        for gpu_id in range(resolved_gpus)
    ]
    
    scheduler = NodeGPUScheduler(resources, max_per_gpu=max_per_gpu)
    
    # 预建立 SSH 连接
    if resolved_nodes and resolved_nodes != ["localhost"]:
        ssh_manager = get_ssh_manager()
        ssh_manager.ensure_connections(resolved_nodes)
    
    # 创建结果表格（线程安全版本）
    # 初始任务数为 0，每个算法运行时更新自己的任务数
    result_table = BatchResultTable(
        algorithms=algorithms,
        datasets=datasets,
        metrics=metrics_list,
        total_tasks=0,  # 会在运行时更新
        primary_metric=primary_metric,
    )
    
    # 创建渲染器
    def _render(clear: bool) -> None:
        result_table.display(clear=clear, consume_improvement=True)
    
    renderer = RenderScheduler(_render)
    
    return SharedResources(
        scheduler=scheduler,
        result_table=result_table,
        task_config=task_config,
        cluster_config=cluster_config,
        renderer=renderer,
        datasets=datasets,
        metrics=metrics_list,
        primary_metric=primary_metric,
    )


def run_with_shared_resources(
    domain: str,
    task: str,
    model: str,
    algorithm_path: str,
    hyperparameter_path: str,
    shared: SharedResources,
    timeout: int = 10800,
    expand_hyperparameters: bool = True,
    output_dir: Path = None,
    save_logs: bool = True,
    fail_fast: bool = True,
) -> Dict[str, Any]:
    """
    使用共享资源运行单个算法
    
    与 run() 的区别：
    - 使用外部传入的共享 GPU 调度器和结果表格
    - 适合多算法并行独立迭代场景
    - 线程安全
    
    Args:
        domain: 领域
        task: 任务
        model: 模型名
        algorithm_path: 算法文件路径
        hyperparameter_path: 超参数文件路径
        shared: 共享资源
        timeout: 单任务超时（秒）
        expand_hyperparameters: 是否展开超参数
        output_dir: 输出目录
        save_logs: 是否保存日志
        fail_fast: 是否遇到首个失败即停止
    
    Returns:
        Dict: 运行结果，包含:
            - success_count: 成功任务数
            - total_tasks: 总任务数
            - best: 最佳结果
            - fatal_error: 致命错误（如果有）
    """
    import threading
    import time
    
    logger = get_logger("interface")
    output_dir = Path(output_dir) if output_dir else Path("output")
    algorithm_path = Path(algorithm_path)
    hyperparameter_path = Path(hyperparameter_path)
    
    task_config = shared.task_config
    cluster_config = shared.cluster_config
    scheduler = shared.scheduler
    result_table = shared.result_table
    renderer = shared.renderer
    datasets = shared.datasets
    
    # 初始化日志
    run_logger = None
    if save_logs:
        run_logger = create_run_logger(model)
    
    # 展开超参数
    hp_files = [hyperparameter_path]
    if expand_hyperparameters:
        try:
            expander = get_expander(domain, task)
            hp_output_dir = output_dir / "hyperparameters" / model
            hp_output_dir.mkdir(parents=True, exist_ok=True)
            hp_files = expander.expand(hyperparameter_path, hp_output_dir)
            logger.info(f"[{model}] Expanded to {len(hp_files)} hyperparameter configs")
        except ValueError:
            logger.warning(f"[{model}] No expander registered, using original hyperparameter file")
    
    # 生成任务列表
    tasks = []
    for dataset in datasets:
        for hp_path in hp_files:
            tasks.append({
                "dataset": dataset,
                "hp_path": hp_path,
                "hp_name": hp_path.stem,
            })
    
    # 更新表格任务数
    result_table.set_algorithm_total_tasks(model, len(tasks))
    
    # 追踪变量
    results: List[TaskResult] = []
    success_count = 0
    fatal_result: Optional[TaskResult] = None
    stop_event = threading.Event()
    
    tasks_per_dataset = {ds: 0 for ds in datasets}
    for t in tasks:
        tasks_per_dataset[t["dataset"]] += 1
    completed_by_dataset = {ds: 0 for ds in datasets}
    
    use_mmrec_stream = (task_config.framework or "").lower() == "mmrec"
    
    from .collector.mmrec_collector import MMRecCollector
    
    def run_single_task(task_info: dict, resource: Optional[GPUResource], task_stop_event: threading.Event) -> TaskResult:
        """执行单个任务"""
        dataset = task_info["dataset"]
        hp_path = task_info["hp_path"]
        hp_name = task_info["hp_name"]
        
        node = resource.node if resource else "localhost"
        gpu_id = resource.gpu_id if resource else 0
        
        stream_collector = MMRecCollector() if use_mmrec_stream else None
        
        # 标记数据集开始运行
        result_table.mark_running(model, dataset)
        renderer.request(clear=True)
        
        if run_logger:
            run_logger.log_task_start(dataset, hp_name, node, gpu_id)
        
        start_time = time.time()
        
        def make_line_callback(log_callback=None):
            def _callback(line: str, is_stderr: bool) -> None:
                if log_callback:
                    log_callback(line, is_stderr)
                if is_stderr or stream_collector is None:
                    return
                metrics = stream_collector.feed(line)
                if metrics:
                    improvement = result_table.update(model, dataset, metrics, hp_name, mark_done=False)
                    if improvement:
                        renderer.request(clear=True)
            return _callback
        
        try:
            # 构建挂载路径
            bind_paths = build_bind_paths(
                task_config=task_config,
                shared_root=cluster_config.shared_root,
                model=model,
                dataset=dataset,
                algorithm_path=algorithm_path,
                hyperparameter_path=hp_path,
                log_dir=run_logger.tasks_dir if run_logger else None,
            )
            
            # 创建执行器
            sif_path = str(Path(cluster_config.shared_root) / "paperbench_pro" / task_config.execution.container)
            
            executor = SSHSingularityExecutor(
                sif_path=sif_path,
                bind_paths=bind_paths,
                node=node,
                gpu_id=gpu_id,
                singularity_exe=cluster_config.singularity.executable if cluster_config.singularity else "singularity",
            )
            
            # 构建命令
            workdir = task_config.execution.workdir if task_config.execution else "/app"
            
            if task_config.args and "template" in task_config.args:
                args_str = task_config.args["template"].format(
                    model=model,
                    dataset=dataset,
                    gpu=gpu_id,
                )
                cmd = ["python", "replicate.py"] + args_str.split()
            else:
                entry_str = task_config.execution.entry.format(
                    model=model,
                    dataset=dataset,
                    gpu=gpu_id,
                )
                cmd = entry_str.split()
            
            # 执行
            if run_logger:
                with run_logger.stream_task_output(dataset, hp_name) as log_callback:
                    line_callback = make_line_callback(log_callback)
                    result = executor.execute(
                        command=cmd,
                        work_dir=workdir,
                        timeout=timeout,
                        streaming=False,
                        line_callback=line_callback,
                        stop_event=task_stop_event,
                    )
            else:
                line_callback = make_line_callback()
                result = executor.execute(
                    command=cmd,
                    work_dir=workdir,
                    timeout=timeout,
                    streaming=False,
                    line_callback=line_callback,
                    stop_event=task_stop_event,
                )
            
            # 解析结果
            metrics = parse_output(
                stdout=result.stdout,
                domain=domain,
                task=task,
            )
            
            success = result.return_code == 0 and len(metrics) > 0
            
            error_msg = None
            if not success:
                if result.return_code != 0:
                    error_msg = f"Exit code {result.return_code}"
                elif len(metrics) == 0:
                    error_msg = "No metrics found in output"
                else:
                    error_msg = "Unknown error"
            
            parsed = TaskResult(
                algorithm=model,
                dataset=dataset,
                hp_name=hp_name,
                metrics=metrics,
                success=success,
                duration=time.time() - start_time,
                error=error_msg,
                raw_stdout=result.stdout if not success else None,
                raw_stderr=result.stderr if not success else None,
                node=node,
                gpu_id=gpu_id,
            )
            
            # 更新表格
            result_table.mark_completed(model, dataset, parsed.success)
            completed_by_dataset[dataset] += 1
            
            if parsed.success and parsed.metrics:
                result_table.update(model, dataset, parsed.metrics, hp_name, mark_done=False)
            
            if completed_by_dataset[dataset] >= tasks_per_dataset[dataset]:
                ds_result = result_table.results.get(model, {}).get(dataset)
                status = "done" if (ds_result and ds_result.fail_count == 0) else "failed"
                result_table.set_status(model, dataset, status)
            
            renderer.request(clear=True)
            
            if run_logger:
                run_logger.log_task_end(dataset, hp_name, parsed.success, parsed.duration)
            
            return parsed
            
        except Exception as e:
            logger.error(f"[{model}] Task failed: {dataset}/{hp_name} - {e}")
            result_table.mark_completed(model, dataset, False)
            completed_by_dataset[dataset] += 1
            
            if completed_by_dataset[dataset] >= tasks_per_dataset[dataset]:
                result_table.set_status(model, dataset, "failed")
            
            renderer.request(clear=True)
            
            parsed = TaskResult(
                algorithm=model,
                dataset=dataset,
                hp_name=hp_name,
                success=False,
                duration=time.time() - start_time,
                error=str(e),
                node=node,
                gpu_id=gpu_id,
            )
            
            if run_logger:
                run_logger.log_task_end(dataset, hp_name, False, parsed.duration)
            
            return parsed
    
    # 使用共享调度器执行任务
    def should_stop(result: TaskResult) -> bool:
        nonlocal fatal_result
        if fail_fast and not result.success:
            fatal_result = result
            return True
        return False
    
    task_results, stopped_result = scheduler.run_batch(
        tasks=tasks,
        run_fn=run_single_task,
        max_workers=min(16, len(tasks)),
        fail_fast=fail_fast,
        should_stop=should_stop,
    )
    
    # 收集结果
    for task_info, result in task_results:
        results.append(result)
        if result.success:
            success_count += 1
    
    if stopped_result:
        result_table.mark_algorithm_stopped(model)
    
    # 构建返回结果
    best_metrics: Dict[str, float] = {}
    best_hp = None
    best_value = 0.0
    
    for result in results:
        if result.success and result.metrics:
            value = result.metrics.get(shared.primary_metric, 0)
            if value > best_value:
                best_value = value
                best_metrics = result.metrics.copy()
                best_hp = result.hp_name
    
    result_dict = {
        "success_count": success_count,
        "total_tasks": len(tasks),
        "best": {
            "metrics": best_metrics,
            "hp_info": best_hp,
        } if best_metrics else {},
        "fatal_error": {
            "stdout": fatal_result.raw_stdout,
            "stderr": fatal_result.raw_stderr,
        } if fatal_result else None,
        "best_per_dataset": {},
    }
    
    # 提取每个数据集的最佳结果
    algo_results = result_table.results.get(model, {})
    for ds, ds_result in algo_results.items():
        if ds_result.metrics:
            result_dict["best_per_dataset"][ds] = {
                "metrics": ds_result.metrics.copy(),
                "hp": ds_result.best_hp,
            }
    
    return result_dict
