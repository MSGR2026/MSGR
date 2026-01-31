"""
ReplicationAgent - 论文复现 Agent

简化版 Agent，复用现有组件：
- PromptBuilder: 构建提示词
- AlgorithmGenerator: 生成算法代码
- HyperparameterGenerator: 生成超参数
- paperbench_pro.interface_v2: 执行评测（直接调用）
- Session: 会话状态管理

功能：
- Round 1: 完整生成代码
- Round 2+: Agent 自主决定是完整重写还是使用 str_replace 修复
"""

from __future__ import annotations

import re
import sys
import time
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# 添加 src 到路径
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.session import Session, SessionManager, SessionStatus, RoundResult
from agent.prompt.prompt_builder import PromptBuilder
from agent.generator.algorithm_generator import AlgorithmGenerator, GenerationError
from agent.generator.hyperparameter_generator import HyperparameterGenerator
from common.logger import create_logger

# 直接使用 PaperBenchPro 接口
from paperbench_pro.interface import run as paperbench_run

if TYPE_CHECKING:
    from graph.local_graph.local_graph_loader import LocalGraphLoader
    from graph.global_graph import GlobalGraph
    from llm.client import LLMClient
    from common.config import Config


@dataclass
class AgentConfig:
    """Agent 配置
    
    简化版：HPC/容器配置由 PaperBenchPro 管理
    """
    max_rounds: int = 10
    
    # 任务配置
    domain: str = "Recsys"
    task_name: str = "MultiModal"
    target_method: str = "BM3_2023"
    target_metric: str = "recall@20"  # 主要评测指标
    
    # 路径配置
    output_dir: str = "output"
    log_dir: str = "logs"
    data_root: str = "ai-scientist/data"
    
    # PaperBenchPro 执行配置（从其自身配置读取）
    max_concurrent: int = 8
    hpc_nodes: Optional[List[str]] = None  # 默认使用 PaperBenchPro 配置
    hpc_gpus_per_node: int = 8
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "AgentConfig":
        """从 YAML 文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            max_rounds=data.get('agent', {}).get('max_rounds', 10),
            domain=data.get('task', {}).get('domain', 'Recsys'),
            task_name=data.get('task', {}).get('task_name', 'MultiModal'),
            target_method=data.get('task', {}).get('target_method', 'BM3_2023'),
            target_metric=data.get('task', {}).get('target_metric', 'recall@20'),
            output_dir=data.get('paths', {}).get('output_dir', 'output'),
            log_dir=data.get('paths', {}).get('log_dir', 'logs'),
            data_root=data.get('paths', {}).get('data_root', 'ai-scientist/data'),
            max_concurrent=data.get('execution', {}).get('max_concurrent', 8),
            hpc_nodes=data.get('execution', {}).get('nodes'),
            hpc_gpus_per_node=data.get('execution', {}).get('gpus_per_node', 8),
        )


@dataclass
class ReplicationResult:
    """复现结果"""
    session_id: str
    paper_id: str
    model_name: str
    success: bool
    best_acc: float
    best_round: int
    total_rounds: int
    final_status: str
    output_dir: str
    duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "paper_id": self.paper_id,
            "model_name": self.model_name,
            "success": self.success,
            "best_acc": self.best_acc,
            "best_round": self.best_round,
            "total_rounds": self.total_rounds,
            "final_status": self.final_status,
            "output_dir": self.output_dir,
            "duration": self.duration,
        }


class ReplicationAgent:
    """
    论文复现 Agent
    
    简化版设计：
    - Round 1: 生成完整代码
    - Round 2+: Agent 自主决定是完整重写还是使用 str_replace 修复
    - 配置统一由 configs/agent.yaml 管理
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        config: Optional[AgentConfig] = None,
        config_path: str = "ai-scientist/configs/agent.yaml",
        local_graph: Optional["LocalGraphLoader"] = None,
        global_graph: Optional["GlobalGraph"] = None,
        global_config: Optional["Config"] = None,
        logger_name: Optional[str] = None,
    ):
        """
        初始化 Agent
        
        Args:
            llm_client: LLM 客户端
            config: Agent 配置（优先使用）
            config_path: 配置文件路径（config 为 None 时使用）
            local_graph: 本地图加载器（可选，会自动创建）
            global_graph: 全局图（可选）
        """
        # 加载配置
        if config is not None:
            self.config = config
        else:
            self.config = AgentConfig.from_yaml(config_path)
        
        self.llm = llm_client
        
        # 初始化 Logger
        resolved_logger_name = logger_name or f"agent_{self.config.domain}_{self.config.task_name}"
        self.logger = create_logger(
            name=resolved_logger_name,
            log_dir=self.config.log_dir,
            use_emoji=True,
        )
        if hasattr(self.llm, "set_logger"):
            self.llm.set_logger(self.logger)
        
        # 初始化 LocalGraph
        if local_graph is not None:
            self.local_graph = local_graph
        else:
            self.local_graph = self._create_local_graph()
        
        # 初始化 GlobalGraph (自动加载)
        if global_graph is not None:
            self.global_graph = global_graph
        else:
            self.global_graph = self._load_global_graph()
        
        # 初始化组件（复用现有）
        self.prompt_builder = PromptBuilder(
            domain=self.config.domain,
            task=self.config.task_name,
            local_graph_loader=self.local_graph,
            global_graph=self.global_graph,
        )
        
        self.algo_generator = AlgorithmGenerator(
            llm_client=self.llm,
            logger=self.logger,
        )
        
        self.hp_generator = HyperparameterGenerator(
            llm_client=self.llm,
            logger=self.logger,
        )
        
        self.session_manager = SessionManager(output_base=self.config.output_dir)
        
        # 保存全局 Config 引用
        self._global_config = global_config
        
        # PaperBenchPro 配置（直接使用 interface_v2）
        self._paperbench_timeout = 10800
        if global_config:
            self._paperbench_timeout = global_config.get("paperbench_pro.timeout", 10800)

    @staticmethod
    def _agent_config_from_global(config: "Config") -> AgentConfig:
        """从全局 Config 构建 AgentConfig，避免读取独立配置文件。"""
        target_method = config.task.paper_id or config.task.task_name
        if isinstance(target_method, list):
            target_method = target_method[0] if target_method else ""
        target_metric = config.get("task.target_metric", "recall@20")
        return AgentConfig(
            max_rounds=config.agent.max_rounds,
            domain=config.task.domain,
            task_name=config.task.task_name,
            target_method=target_method,
            target_metric=target_metric,
            output_dir=config.paths.output_dir,
            log_dir=config.paths.log_dir,
            data_root=config.paths.data_root,
        )

    @classmethod
    def from_config(
        cls,
        config: "Config",
        llm_client: "LLMClient",
        local_graph: Optional["LocalGraphLoader"] = None,
        global_graph: Optional["GlobalGraph"] = None,
        logger_name: Optional[str] = None,
    ) -> "ReplicationAgent":
        """通过统一 Config 创建 Agent 实例。"""
        agent_config = cls._agent_config_from_global(config)
        return cls(
            llm_client=llm_client,
            config=agent_config,
            local_graph=local_graph,
            global_graph=global_graph,
            global_config=config,
            logger_name=logger_name,
        )
    
    def run(self, paper_id: Optional[str] = None) -> ReplicationResult:
        """
        执行论文复现
        
        Args:
            paper_id: 目标论文 ID，默认使用配置中的 target_method
        
        Returns:
            ReplicationResult
        """
        paper_id = paper_id or self.config.target_method
        start_time = time.time()
        
        # 获取模型名
        model_name = self._get_model_name(paper_id)
        
        # 创建 Session
        session = self.session_manager.create_session(
            paper_id=paper_id,
            model_name=model_name,
            domain=self.config.domain,
            task=self.config.task_name,
            max_rounds=self.config.max_rounds,
        )
        
        self.logger.session_start(
            session_id=session.session_id,
            paper_id=paper_id,
            model_name=model_name,
            domain=self.config.domain,
            task=self.config.task_name,
            max_rounds=self.config.max_rounds,
        )
        
        session.start()
        
        # 当前代码（用于传递给下一轮）
        current_algo_code = None
        current_hp_yaml = None
        
        try:
            while not session.should_stop():
                round_num = session.current_round
                self.logger.round_start(round_num, self.config.max_rounds)
                
                # 执行一轮
                round_result, current_algo_code, current_hp_yaml = self._run_round(
                    session, paper_id, current_algo_code, current_hp_yaml
                )
                
                session.update_after_round(round_result)
                self.session_manager.save_session(session)
                
                self.logger.round_end(
                    round_num=round_num,
                    success=round_result.error is None,
                    acc=round_result.acc,
                    error=round_result.error,
                    duration=round_result.duration,
                )
                
        except Exception as e:
            self.logger.error(f"Session error: {e}")
            session.status = SessionStatus.ERROR
        
        session.finish()
        duration = time.time() - start_time
        
        self.logger.session_end(
            success=session.status == SessionStatus.SUCCESS,
            best_acc=session.best_acc,
            best_round=session.best_round,
            total_rounds=session.current_round,
            duration=duration,
            total_tokens=0,
            output_dir=session.output_dir,
        )
        
        # 如果成功，保存最佳结果
        if session.status == SessionStatus.SUCCESS and session.best_round > 0:
            self._save_best_result(session)
        
        return ReplicationResult(
            session_id=session.session_id,
            paper_id=paper_id,
            model_name=model_name,
            success=session.status == SessionStatus.SUCCESS,
            best_acc=session.best_acc,
            best_round=session.best_round,
            total_rounds=session.current_round,
            final_status=session.status.value,
            output_dir=session.output_dir,
            duration=duration,
        )

    def _generate_round_assets(
        self,
        session: Session,
        paper_id: str,
        current_algo_code: Optional[str],
        current_hp_yaml: Optional[str],
    ) -> Tuple[Optional[RoundResult], str, str, Path, Path]:
        """
        生成一轮的算法与超参数文件（不执行评测）。
        
        Returns:
            (round_result_or_none, algo_code, hp_yaml, algo_path, hp_path)
        """
        round_start = time.time()
        round_num = session.current_round
        output_dir = self.session_manager.ensure_round_output_dir(session)
        algo_path = output_dir / "algorithm.py"
        hp_path = output_dir / "hyperparameter.yaml"
        
        try:
            # Step 1: 生成/修复算法代码
            self.logger.step(1, 3, "生成算法代码")
            system_prompt, user_prompt = self.prompt_builder.build_algorithm_prompt(
                paper_id=paper_id,
                history=session.history,
                current_code=current_algo_code,  # Round 2+ 传入当前代码
            )
            
            response = self.algo_generator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                domain=self.config.domain,
            )
            
            # 检查响应是 skip/str_replace 还是完整代码
            algo_code_result = self._process_response(response, current_algo_code)
            
            if algo_code_result is None:
                # Agent 选择不修改算法代码
                if current_algo_code is None:
                    # 第一轮不允许 skip
                    self.logger.error("  第一轮不能跳过算法生成")
                    return RoundResult(
                        round=round_num,
                        algorithm_path=str(algo_path),
                        hyperparameter_path=str(hp_path),
                        error="第一轮不能跳过算法生成",
                        feedback="第一轮必须生成完整的算法代码",
                        duration=time.time() - round_start,
                    ), current_algo_code or "", current_hp_yaml or "", algo_path, hp_path
                
                # 使用上一轮的代码
                algo_code = current_algo_code
                self.logger.info(f"  ✓ Agent 选择保持算法代码不变")
            else:
                algo_code = algo_code_result
                algo_path.write_text(algo_code, encoding='utf-8')
                self.logger.info(f"  算法代码已保存: {algo_path}")
            
            # Step 2: 生成/修复超参数
            self.logger.step(2, 3, "生成超参数配置")
            system_prompt, user_prompt = self.prompt_builder.build_hyperparameter_prompt(
                paper_id=paper_id,
                algorithm_code=algo_code,
                history=session.history,
                current_hyperparameter=current_hp_yaml,  # Round 2+ 传入当前配置
            )
            
            response = self.hp_generator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            # 检查响应是 skip/str_replace 还是完整配置
            hp_yaml_result = self._process_response(response, current_hp_yaml, is_yaml=True)
            
            if hp_yaml_result is None:
                # Agent 选择不修改超参数配置
                if current_hp_yaml is None:
                    # 第一轮不允许 skip
                    self.logger.error("  第一轮不能跳过超参数生成")
                    return RoundResult(
                        round=round_num,
                        algorithm_path=str(algo_path),
                        hyperparameter_path=str(hp_path),
                        error="第一轮不能跳过超参数生成",
                        feedback="第一轮必须生成完整的超参数配置",
                        duration=time.time() - round_start,
                    ), current_algo_code or "", current_hp_yaml or "", algo_path, hp_path
                
                # 使用上一轮的配置
                hp_yaml = current_hp_yaml
                self.logger.info(f"  ✓ Agent 选择保持超参数配置不变")
            else:
                hp_yaml = hp_yaml_result
                hp_path.write_text(hp_yaml, encoding='utf-8')
                self.logger.info(f"  超参数配置已保存: {hp_path}")
            
            # 如果算法代码或超参数有更新，写入文件
            if algo_code_result is not None or hp_yaml_result is not None:
                # 确保文件存在（即使没有修改也要写入，保证路径有效）
                if algo_code_result is None:
                    algo_path.write_text(algo_code, encoding='utf-8')
                if hp_yaml_result is None:
                    hp_path.write_text(hp_yaml, encoding='utf-8')
            # # 无论是否修改，都确保本轮两个文件落盘（保证 PaperBench 路径有效）
            # algo_path.write_text(algo_code, encoding="utf-8")
            # hp_path.write_text(hp_yaml, encoding="utf-8")
            self.logger.info(f"[DEBUG] algo_path_abs={algo_path.resolve()} exists={algo_path.exists()} size={algo_path.stat().st_size if algo_path.exists() else -1}")
            self.logger.info(f"[DEBUG] hp_path_abs={hp_path.resolve()} exists={hp_path.exists()} size={hp_path.stat().st_size if hp_path.exists() else -1}")
            self.logger.info(f"[DEBUG] cwd={Path.cwd()}")


            return None, algo_code, hp_yaml, algo_path, hp_path
        
        except GenerationError as e:
            self.logger.error(f"Generation failed: {e}")
            return RoundResult(
                round=round_num,
                algorithm_path=str(algo_path),
                hyperparameter_path=str(hp_path),
                error=str(e),
                feedback=f"代码生成失败: {e}",
                duration=time.time() - round_start,
            ), current_algo_code or "", current_hp_yaml or "", algo_path, hp_path
        
        except Exception as e:
            self.logger.error(f"Round asset generation error: {e}")
            return RoundResult(
                round=round_num,
                algorithm_path=str(algo_path),
                hyperparameter_path=str(hp_path),
                error=str(e),
                feedback=f"代码生成异常: {e}",
                duration=time.time() - round_start,
            ), current_algo_code or "", current_hp_yaml or "", algo_path, hp_path
    
    def _run_round(
        self,
        session: Session,
        paper_id: str,
        current_algo_code: Optional[str],
        current_hp_yaml: Optional[str],
    ) -> Tuple[RoundResult, str, str]:
        """
        执行一轮复现
        
        - Round 1 (current_algo_code 为 None): 完整生成
        - Round 2+ (current_algo_code 不为 None): Agent 自主决定修复方式
        
        Returns:
            (RoundResult, updated_algo_code, updated_hp_yaml)
        """
        round_start = time.time()
        round_num = session.current_round
        output_dir = self.session_manager.ensure_round_output_dir(session)
        algo_path = output_dir / "algorithm.py"
        hp_path = output_dir / "hyperparameter.yaml"
        
        try:
            round_result, algo_code, hp_yaml, algo_path, hp_path = self._generate_round_assets(
                session=session,
                paper_id=paper_id,
                current_algo_code=current_algo_code,
                current_hp_yaml=current_hp_yaml,
            )
            if round_result is not None:
                return round_result, algo_code, hp_yaml
            
            # Step 3: 执行评测
            self.logger.step(3, 3, "执行 PaperBench-Pro 评测")
            best_by_dataset: Dict[str, Dict[str, float]] = {}
            metrics_order: List[str] = []
            done_by_dataset: Dict[str, int] = {}
            total_by_dataset: Dict[str, int] = {}
            try:
                from paperbench_pro import PaperBenchPro
                domain_cfg = PaperBenchPro.get_domain_config(self.config.domain, self.config.task_name)
                metrics_order = list(domain_cfg.metrics or [])
            except Exception:
                metrics_order = []

            def _format_current_best() -> Optional[str]:
                if not best_by_dataset:
                    return None
                parts = []
                for ds in sorted(best_by_dataset.keys()):
                    metrics = best_by_dataset[ds]
                    order = metrics_order or sorted(metrics.keys())
                    metric_parts = [f"{k}={metrics[k]:.4f}" for k in order if k in metrics]
                    if metric_parts:
                        done = done_by_dataset.get(ds)
                        total = total_by_dataset.get(ds)
                        if done is not None or total is not None:
                            done_str = "?" if done is None else str(done)
                            total_str = "?" if total is None else str(total)
                            parts.append(f"{ds}({done_str}/{total_str}) " + " ".join(metric_parts))
                        else:
                            parts.append(f"{ds} " + " ".join(metric_parts))
                if not parts:
                    return None
                return "Current Best: " + " | ".join(parts)

            def progress_callback(line: str) -> None:
                stripped = line.strip()
                # DEBUG: 打印所有 DEBUG 行
                if stripped.startswith("DEBUG_"):
                    self.logger.info(f"[DEBUG] {stripped}")
                best_idx = stripped.find("MMREC_BEST")
                status_idx = stripped.find("MMREC_STATUS")
                if best_idx == -1 and status_idx == -1:
                    return
                if best_idx != -1 and (status_idx == -1 or best_idx <= status_idx):
                    mmrec_line = stripped[best_idx:]
                    is_best = True
                else:
                    mmrec_line = stripped[status_idx:]
                    is_best = False

                if not is_best:
                    payload = {}
                    for chunk in mmrec_line.split()[1:]:
                        if "=" in chunk:
                            key, value = chunk.split("=", 1)
                            payload[key] = value
                    event = payload.get("event")
                    if event == "init":
                        self.logger.info(
                            "MMRec init: tasks=%s workers=%s gpus=%s",
                            payload.get("total_tasks", "?"),
                            payload.get("workers", "?"),
                            payload.get("gpus", "?"),
                        )
                    elif event == "dataset_total":
                        dataset = payload.get("dataset")
                        total = payload.get("total")
                        if dataset and total and total.isdigit():
                            total_by_dataset[dataset] = int(total)
                            done_by_dataset.setdefault(dataset, 0)
                    elif event == "done":
                        self.logger.info(
                            "MMRec done: %s %s h%s code=%s gpu=%s",
                            payload.get("dataset", "?"),
                            payload.get("model", "?"),
                            payload.get("combo", "?"),
                            payload.get("returncode", "?"),
                            payload.get("gpu", ""),
                        )
                        dataset = payload.get("dataset")
                        done = payload.get("done")
                        if dataset and done and done.isdigit():
                            done_by_dataset[dataset] = int(done)
                    return

                payload = {}
                for chunk in mmrec_line.split()[1:]:
                    if "=" in chunk:
                        key, value = chunk.split("=", 1)
                        payload[key] = value
                dataset = payload.pop("dataset", None)
                if not dataset:
                    return
                done_val = payload.pop("done", None)
                total_val = payload.pop("total", None)
                if done_val and done_val.isdigit():
                    done_by_dataset[dataset] = int(done_val)
                if total_val and total_val.isdigit():
                    total_by_dataset[dataset] = int(total_val)
                metrics: Dict[str, float] = {}
                for key, value in payload.items():
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        continue
                if not metrics:
                    return
                best_by_dataset[dataset] = metrics
                formatted = _format_current_best()
                if formatted:
                    self.logger.info(formatted)

            # 直接调用 PaperBenchPro 新版接口
            model_name = self._get_model_name(paper_id)
            result_dict = paperbench_run(
                domain=self.config.domain,
                task=self.config.task_name,
                model=model_name,
                algorithm_path=str(algo_path),
                hyperparameter_path=str(hp_path),
                timeout=self._paperbench_timeout,
            )
            
            duration = time.time() - round_start
            
            # 使用 RoundResult.from_paperbench 转换结果
            round_result = RoundResult.from_paperbench(
                data=result_dict,
                round_num=round_num,
                algorithm_path=str(algo_path),
                hyperparameter_path=str(hp_path),
                primary_metric=self.config.target_metric,
                duration=duration,
            )
            
            # 保存原始 PaperBench 数据（用于后续提取 combo 信息）
            round_result.raw_paperbench_data = result_dict
            
            return round_result, algo_code, hp_yaml
        
        except Exception as e:
            self.logger.error(f"Round execution failed: {e}")
            return RoundResult(
                round=round_num,
                algorithm_path=str(algo_path),
                hyperparameter_path=str(hp_path),
                error=str(e),
                feedback=f"执行失败: {e}",
                duration=time.time() - round_start,
            ), current_algo_code or "", current_hp_yaml or ""
            
    def _round_result_from_batch(
        self,
        batch_result: Dict[str, Any],
        algorithm_name: str,
        round_num: int,
        algorithm_path: str,
        hyperparameter_path: str,
        duration: float,
    ) -> RoundResult:
        """将 batch 结果转换为单轮 RoundResult。"""
        results_by_algorithm = batch_result.get("results_by_algorithm", {})
        algo_results = results_by_algorithm.get(algorithm_name, []) or []
        success_count = sum(1 for r in algo_results if r.get("success"))
        total_tasks = len(algo_results)
        
        best_metrics: Dict[str, float] = {}
        best_value = float("-inf")
        best_by_algorithm = batch_result.get("best_by_algorithm", {})
        for ds_info in (best_by_algorithm.get(algorithm_name, {}) or {}).values():
            metrics = ds_info.get("metrics") or {}
            value = metrics.get(self.config.target_metric, 0.0)
            if metrics and value >= best_value:
                best_value = value
                best_metrics = metrics
        
        if not best_metrics:
            for result in algo_results:
                metrics = result.get("metrics") or {}
                value = metrics.get(self.config.target_metric, 0.0)
                if metrics and value >= best_value:
                    best_value = value
                    best_metrics = metrics
        
        if not best_metrics:
            tracking = batch_result.get("realtime_tracking", {})
            algo_tracking = tracking.get("algorithms", {}).get(algorithm_name, {})
            for ds_info in (algo_tracking.get("datasets", {}) or {}).values():
                metrics = ds_info.get("metrics") or {}
                value = metrics.get(self.config.target_metric, 0.0)
                if metrics and value >= best_value:
                    best_value = value
                    best_metrics = metrics
        
        fatal = batch_result.get("fatal_by_algorithm", {}).get(algorithm_name)
        fatal_error = None
        if fatal:
            if isinstance(fatal, dict):
                fatal_error = {
                    "stdout": fatal.get("raw_stdout") or fatal.get("stdout"),
                    "stderr": fatal.get("raw_stderr") or fatal.get("stderr"),
                }
            else:
                fatal_error = {"stdout": None, "stderr": None}
        
        result_dict = {
            "success_count": success_count,
            "total_tasks": total_tasks,
            "best": {"metrics": best_metrics} if best_metrics else {},
            "fatal_error": fatal_error,
        }
        
        return RoundResult.from_paperbench(
            data=result_dict,
            round_num=round_num,
            algorithm_path=algorithm_path,
            hyperparameter_path=hyperparameter_path,
            primary_metric=self.config.target_metric,
            duration=duration,
        )

    def _process_response(
        self,
        response: str,
        current_content: Optional[str],
        is_yaml: bool = False,
    ) -> Optional[str]:
        """
        处理 LLM 响应：检测是 tool_call (skip/str_replace) 还是完整内容
        
        Args:
            response: LLM 响应
            current_content: 当前内容（用于 str_replace 或 skip）
            is_yaml: 是否是 YAML 格式
        
        Returns:
            处理后的内容，如果是 skip 则返回 None
        """
        # 检查是否包含 skip 工具调用
        skip_algo_pattern = r'<tool_call>\s*<name>skip_algorithm</name>\s*</tool_call>'
        skip_hp_pattern = r'<tool_call>\s*<name>skip_hyperparameter</name>\s*</tool_call>'
        
        if re.search(skip_algo_pattern, response) or re.search(skip_hp_pattern, response):
            tool_name = "skip_algorithm" if re.search(skip_algo_pattern, response) else "skip_hyperparameter"
            self.logger.info(f"  Agent 选择跳过修改 ({tool_name})")
            return None  # 返回 None 表示不修改
        
        # 检查是否包含 XML 格式的 tool_call (新格式)
        if "<tool_call>" in response and "<name>str_replace</name>" in response:
            if current_content is None:
                self.logger.warning("收到 str_replace 工具调用但无当前内容，使用完整响应")
                return self._extract_code_block(response, is_yaml)
            
            # 应用 XML 格式的 str_replace
            return self._apply_str_replace_xml(response, current_content)
        
        # 兼容旧格式：检查是否包含 <<<old>>> 和 <<<new>>> 指令
        if "<<<old>>>" in response and "<<<new>>>" in response:
            if current_content is None:
                self.logger.warning("收到 str_replace 但无当前内容，使用完整响应")
                return self._extract_code_block(response, is_yaml)
            
            # 应用旧格式的 str_replace
            return self._apply_str_replace(response, current_content)
        
        # 提取代码块
        return self._extract_code_block(response, is_yaml)
    
    def _apply_str_replace_xml(self, response: str, current_content: str) -> str:
        """
        应用 XML 格式的 str_replace 工具调用
        
        支持格式:
        <tool_call>
        <name>str_replace</name>
        <old_str>旧内容</old_str>
        <new_str>新内容</new_str>
        </tool_call>
        """
        # 解析 tool_call 块
        pattern = r'<tool_call>\s*<name>str_replace</name>\s*<old_str>\s*(.*?)\s*</old_str>\s*<new_str>\s*(.*?)\s*</new_str>\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if not matches:
            self.logger.warning("未找到有效的 str_replace 工具调用，返回原内容")
            return current_content
        
        self.logger.info(f"  找到 {len(matches)} 个 str_replace 工具调用")
        
        updated_content = current_content
        for i, (old_str, new_str) in enumerate(matches):
            # 保留内容的原始格式，只去除首尾空白行
            old_str = old_str.strip('\n')
            new_str = new_str.strip('\n')
            
            if old_str in updated_content:
                updated_content = updated_content.replace(old_str, new_str, 1)
                self.logger.info(f"  [{i+1}] str_replace: {len(old_str)} chars -> {len(new_str)} chars ✓")
            else:
                self.logger.warning(f"  [{i+1}] str_replace: 未找到匹配的旧内容 ✗")
                self.logger.debug(f"      old_str 前50字符: {repr(old_str[:50])}")
        
        return updated_content
    
    def _apply_str_replace(self, response: str, current_content: str) -> str:
        """
        应用旧格式的 str_replace 修改（兼容）
        
        支持格式:
        ```str_replace
        <<<old>>>
        旧内容
        <<<new>>>
        新内容
        ```
        """
        # 解析 str_replace 块
        pattern = r'```str_replace\s*\n<<<old>>>\s*\n(.*?)<<<new>>>\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if not matches:
            # 尝试不带 ```str_replace 的格式
            pattern2 = r'<<<old>>>\s*\n(.*?)<<<new>>>\s*\n(.*?)(?=<<<|$)'
            matches = re.findall(pattern2, response, re.DOTALL)
        
        if not matches:
            self.logger.warning("未找到有效的 str_replace 指令，返回原内容")
            return current_content
        
        updated_content = current_content
        for old_str, new_str in matches:
            old_str = old_str.strip()
            new_str = new_str.strip()
            
            if old_str in updated_content:
                updated_content = updated_content.replace(old_str, new_str, 1)
                self.logger.info(f"  str_replace: {len(old_str)} chars -> {len(new_str)} chars")
            else:
                self.logger.warning(f"  str_replace: 未找到匹配的旧内容")
        
        return updated_content
    
    def _extract_code_block(self, response: str, is_yaml: bool = False) -> str:
        """
        从响应中提取代码块
        """
        lang = "yaml" if is_yaml else "python"
        
        # 尝试提取 ```python 或 ```yaml 代码块
        pattern = rf'```{lang}\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 尝试提取通用代码块
        pattern = r'```\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # 返回原响应
        return response.strip()
    
    def _create_local_graph(self) -> "LocalGraphLoader":
        """创建 LocalGraphLoader"""
        from graph.local_graph.local_graph_loader import LocalGraphLoader
        return LocalGraphLoader(
            domain=self.config.domain,
            task=self.config.task_name,
            data_root=Path(self.config.data_root),
        )
    
    def _load_global_graph(self):
        """加载 GlobalGraph"""
        from graph.global_graph.loader import GlobalGraphLoader
        from graph.global_graph.data_structures import GlobalGraph
        
        try:
            loader = GlobalGraphLoader(data_root=str(self.config.data_root))
            graph = loader.load(
                domain=self.config.domain,
                task=self.config.task_name,
            )
            
            if graph is not None:
                self.logger.info(f"✅ GlobalGraph 已加载 ({len(graph.items)} 条知识)")
            else:
                self.logger.info("ℹ️ GlobalGraph 未找到或为空")
            
            return graph
        except Exception as e:
            self.logger.warning(f"⚠️ GlobalGraph 加载失败: {e}")
            return None
    
    # HPC 执行器已移至 PaperBenchPro (paperbench_pro/executor/)
    # Agent 现在直接调用 paperbench_pro.interface_v2.run()
    
    def _get_model_name(self, paper_id: str) -> str:
        """从 paper_id 获取模型名"""
        try:
            info = self.local_graph.get_paper_info(paper_id)
            if info and "alias" in info:
                return info["alias"]
        except:
            pass
        
        # Fallback: BM3_2023 -> BM3
        parts = paper_id.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return paper_id
    
    def _save_best_result(self, session: Session) -> None:
        """
        保存最佳结果到 best/ 目录（成功时调用）
        
        保存内容：
        - algorithm.py: 最佳轮次的算法代码
        - hyperparameter.yaml: 最佳轮次的超参数模板
        - best_combo.yaml: 最佳超参数组合（从 output/hyperparameters/ 复制）
        - success.json: 成功信息（包含最佳 combo 名称和指标）
        """
        try:
            import shutil
            import json
            
            # 1. 创建 best 目录
            best_dir = Path(session.output_dir) / "best"
            best_dir.mkdir(exist_ok=True)
            
            # 2. 找到最佳轮次的文件
            best_round_dir = Path(session.output_dir) / f"round_{session.best_round}"
            
            # 3. 复制算法代码和超参数模板
            for filename in ["algorithm.py", "hyperparameter.yaml"]:
                src = best_round_dir / filename
                if src.exists():
                    shutil.copy2(src, best_dir / filename)
                    self.logger.info(f"  ✓ 已复制: {filename}")
                else:
                    self.logger.warning(f"  ⚠ 文件不存在: {src}")
            
            # 4. 获取最佳轮次的结果（包含 combo 信息）
            best_result = session.history[session.best_round - 1]
            
            # 5. 从 PaperBench 结果中提取最佳 combo 信息
            best_combo_info = self._extract_best_combo_info(best_result)
            best_combo_name = best_combo_info.get("combo_name")
            
            # 6. 如果找到最佳 combo，复制对应的 combo 文件
            if best_combo_name:
                combo_src = Path(self.config.output_dir) / "hyperparameters" / session.model_name / f"{best_combo_name}.yaml"
                if combo_src.exists():
                    shutil.copy2(combo_src, best_dir / "best_combo.yaml")
                    self.logger.info(f"  ✓ 已复制最佳超参数: {best_combo_name}")
                else:
                    self.logger.warning(f"  ⚠ 未找到 combo 文件: {combo_src}")
            else:
                self.logger.warning(f"  ⚠ 未能提取最佳 combo 信息")
            
            # 7. 保存成功信息（包含 combo 信息和详细指标）
            success_info = {
                "success": True,
                "paper_id": session.paper_id,
                "model_name": session.model_name,
                "domain": session.domain,
                "task": session.task,
                "best_round": session.best_round,
                "best_acc": session.best_acc,
                "best_combo": best_combo_name,
                "best_combo_info": best_combo_info,
                "all_metrics": best_result.metrics,
                "timestamp": datetime.now().isoformat(),
                "total_rounds": session.current_round,
                "duration": session.get_total_duration(),
            }
            
            success_file = best_dir / "success.json"
            with open(success_file, "w", encoding="utf-8") as f:
                json.dump(success_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 最佳结果已保存到: {best_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠ 保存最佳结果失败: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _extract_best_combo_info(self, result: RoundResult) -> Dict[str, Any]:
        """
        从 RoundResult 中提取最佳 combo 信息
        
        PaperBench 返回的数据结构：
        - result_dict["best"]["hp_info"]: overall best 的 combo 名称
        - result_dict["best_per_dataset"][ds]["hp"]: 每个数据集的最佳 combo
        - result_dict["realtime_tracking"]: 实时追踪信息
        
        Returns:
            Dict 包含:
            - combo_name: str, overall best combo 名称（如 "combo_5"）
            - by_dataset: Dict[str, str], 每个数据集的最佳 combo
            - best_metrics: Dict[str, float], overall best 的指标
        """
        combo_info = {
            "combo_name": None,
            "by_dataset": {},
            "best_metrics": {},
        }
        
        # 如果有原始 PaperBench 数据，从中提取
        if result.raw_paperbench_data:
            data = result.raw_paperbench_data
            
            # 提取 overall best combo
            if "best" in data and data["best"]:
                best = data["best"]
                combo_info["combo_name"] = best.get("hp_info")
                combo_info["best_metrics"] = best.get("metrics", {})
                if combo_info["combo_name"]:
                    self.logger.debug(f"  从 raw_data 提取到 overall best combo: {combo_info['combo_name']}")
            
            # 提取每个数据集的最佳 combo
            if "best_per_dataset" in data:
                for ds, ds_info in data["best_per_dataset"].items():
                    hp = ds_info.get("hp")
                    if hp:
                        combo_info["by_dataset"][ds] = hp
                if combo_info["by_dataset"]:
                    self.logger.debug(f"  从 raw_data 提取到 {len(combo_info['by_dataset'])} 个数据集的 best combo")
            
            # 如果 best 为空，尝试从 realtime_tracking 提取
            if not combo_info["combo_name"] and "realtime_tracking" in data:
                tracking = data["realtime_tracking"]
                # 尝试从 best_by_dataset 或 datasets 中找到最佳的
                datasets_info = tracking.get("best_by_dataset") or tracking.get("datasets", {})
                best_value = 0.0
                best_hp = None
                primary_metric = self.config.target_metric
                
                for ds, ds_info in datasets_info.items():
                    metrics = ds_info.get("metrics", {})
                    hp = ds_info.get("best_hp") or ds_info.get("hp")
                    value = metrics.get(primary_metric, 0.0)
                    if value > best_value:
                        best_value = value
                        best_hp = hp
                        combo_info["best_metrics"] = metrics
                
                if best_hp:
                    combo_info["combo_name"] = best_hp
                    self.logger.debug(f"  从 realtime_tracking 提取到 best combo: {combo_info['combo_name']}")
        
        # Fallback: 从 feedback 中解析
        if not combo_info["combo_name"] and result.feedback:
            import re
            match = re.search(r'\[?(combo_\d+)\]?', result.feedback)
            if match:
                combo_info["combo_name"] = match.group(1)
                self.logger.debug(f"  从 feedback 提取到 combo: {combo_info['combo_name']}")
        
        if not combo_info["combo_name"]:
            self.logger.debug("  无法提取 best combo 信息")
        
        return combo_info


def create_agent(
    llm_client: "LLMClient",
    config_path: str = "ai-scientist/configs/agent.yaml",
    logger_name: Optional[str] = None,
) -> ReplicationAgent:
    """
    创建 Agent 的便捷函数
    
    Args:
        llm_client: LLM 客户端
        config_path: 配置文件路径
    
    Returns:
        ReplicationAgent 实例
    """
    return ReplicationAgent(
        llm_client=llm_client,
        config_path=config_path,
        logger_name=logger_name,
    )
