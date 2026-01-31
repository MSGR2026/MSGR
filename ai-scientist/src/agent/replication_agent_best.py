"""
Best-only ReplicationAgent.

Loads existing best artifacts and runs PaperBenchPro on a specified GPU set.
No LLM-based code generation is used.
"""
from __future__ import annotations

import sys
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.session import SessionManager, SessionStatus, RoundResult
from common.logger import create_logger
from paperbench_pro.interface import create_shared_resources, run_with_shared_resources
from paperbench_pro.executor import get_ssh_manager
from paperbench_pro.scheduler import GPUResource, NodeGPUScheduler

if TYPE_CHECKING:
    from graph.local_graph.local_graph_loader import LocalGraphLoader
    from graph.global_graph import GlobalGraph
    from common.config import Config


@dataclass
class AgentConfig:
    """Agent config (best-only)."""

    max_rounds: int = 1

    # Task
    domain: str = "Recsys"
    task_name: str = "MultiModal"
    target_method: str = "BM3_2023"
    target_metric: str = "recall@20"

    # Paths
    output_dir: str = "output"
    log_dir: str = "logs"
    data_root: str = "ai-scientist/data"

    # Execution
    max_concurrent: int = 8
    hpc_nodes: Optional[List[str]] = None
    hpc_gpus_per_node: Optional[int] = None
    gpu_ids: Optional[List[int]] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "AgentConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            max_rounds=data.get("agent", {}).get("max_rounds", 1),
            domain=data.get("task", {}).get("domain", "Recsys"),
            task_name=data.get("task", {}).get("task_name", "MultiModal"),
            target_method=data.get("task", {}).get("target_method", "BM3_2023"),
            target_metric=data.get("task", {}).get("target_metric", "recall@20"),
            output_dir=data.get("paths", {}).get("output_dir", "output"),
            log_dir=data.get("paths", {}).get("log_dir", "logs"),
            data_root=data.get("paths", {}).get("data_root", "ai-scientist/data"),
            max_concurrent=data.get("execution", {}).get("max_concurrent", 8),
            hpc_nodes=data.get("execution", {}).get("nodes"),
            hpc_gpus_per_node=data.get("execution", {}).get("gpus_per_node"),
        )


@dataclass
class ReplicationResult:
    """Replication result."""

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
    """Best-only replication agent using existing outputs."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        config_path: str = "ai-scientist/configs/agent.yaml",
        local_graph: Optional["LocalGraphLoader"] = None,
        global_graph: Optional["GlobalGraph"] = None,
        global_config: Optional["Config"] = None,
        logger_name: Optional[str] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = AgentConfig.from_yaml(config_path)

        resolved_logger_name = logger_name or f"agent_best_{self.config.domain}_{self.config.task_name}"
        self.logger = create_logger(
            name=resolved_logger_name,
            log_dir=self.config.log_dir,
            use_emoji=True,
        )

        if local_graph is not None:
            self.local_graph = local_graph
        else:
            self.local_graph = self._create_local_graph()

        if global_graph is not None:
            self.global_graph = global_graph
        else:
            self.global_graph = self._load_global_graph()

        self.session_manager = SessionManager(output_base=self.config.output_dir)

        self._global_config = global_config
        self._paperbench_timeout = 3600
        if global_config:
            self._paperbench_timeout = global_config.get("paperbench_pro.timeout", 3600)

    @staticmethod
    def _agent_config_from_global(config: "Config") -> AgentConfig:
        target_method = config.task.paper_id or config.task.task_name
        if isinstance(target_method, list):
            target_method = target_method[0] if target_method else ""
        target_metric = config.get("task.target_metric", "recall@20")
        return AgentConfig(
            max_rounds=1,
            domain=config.task.domain,
            task_name=config.task.task_name,
            target_method=target_method,
            target_metric=target_metric,
            output_dir=config.paths.output_dir,
            log_dir=config.paths.log_dir,
            data_root=config.paths.data_root,
            max_concurrent=config.get("execution.max_concurrent", 8),
            hpc_nodes=config.get("execution.nodes"),
            hpc_gpus_per_node=config.get("execution.gpus_per_node"),
            gpu_ids=config.resources.gpu_ids,
        )

    @classmethod
    def from_config(
        cls,
        config: "Config",
        local_graph: Optional["LocalGraphLoader"] = None,
        global_graph: Optional["GlobalGraph"] = None,
        logger_name: Optional[str] = None,
    ) -> "ReplicationAgent":
        agent_config = cls._agent_config_from_global(config)
        return cls(
            config=agent_config,
            local_graph=local_graph,
            global_graph=global_graph,
            global_config=config,
            logger_name=logger_name,
        )

    def run(
        self,
        paper_id: Optional[str] = None,
        prefer_best_outputs: bool = True,
        fallback_to_latest: bool = True,
    ) -> ReplicationResult:
        paper_id = paper_id or self.config.target_method
        start_time = time.time()

        model_name = self._get_model_name(paper_id)

        session = self.session_manager.create_session(
            paper_id=paper_id,
            model_name=model_name,
            domain=self.config.domain,
            task=self.config.task_name,
            max_rounds=1,
        )

        self.logger.session_start(
            session_id=session.session_id,
            paper_id=paper_id,
            model_name=model_name,
            domain=self.config.domain,
            task=self.config.task_name,
            max_rounds=1,
        )

        session.start()

        try:
            algo_path, hp_path, expand_hyperparameters = self._resolve_assets(
                model_name=model_name,
                prefer_best_outputs=prefer_best_outputs,
                fallback_to_latest=fallback_to_latest,
            )
            round_num = session.current_round
            self.logger.round_start(round_num, 1)
            round_result = self._run_round_from_assets(
                paper_id=paper_id,
                round_num=round_num,
                algo_path=algo_path,
                hp_path=hp_path,
                expand_hyperparameters=expand_hyperparameters,
            )
            session.update_after_round(round_result)
            session.status = SessionStatus.SUCCESS if round_result.success else SessionStatus.ERROR
            self.session_manager.save_session(session)
            self.logger.round_end(
                round_num=round_num,
                success=round_result.error is None,
                acc=round_result.acc,
                error=round_result.error,
                duration=round_result.duration,
            )
        except Exception as exc:
            self.logger.error(f"Session error: {exc}")
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

    def _resolve_assets(
        self,
        model_name: str,
        prefer_best_outputs: bool,
        fallback_to_latest: bool,
    ) -> Tuple[Path, Path, bool]:
        if prefer_best_outputs:
            best_assets = self._find_best_assets(model_name)
            if best_assets is not None:
                return best_assets
        if fallback_to_latest:
            latest = self._find_latest_round_assets(model_name)
            if latest is not None:
                algo_path, hp_path = latest
                return algo_path, hp_path, True
        raise FileNotFoundError(
            f"No outputs found for {model_name} in {self.config.output_dir}."
        )

    def _find_best_assets(self, model_name: str) -> Optional[Tuple[Path, Path, bool]]:
        base_dir = Path(self.config.output_dir) / self.config.domain / self.config.task_name / model_name
        best_dir = base_dir / "best"
        if not best_dir.exists():
            return None
        algo_path = best_dir / "algorithm.py"
        if not algo_path.exists():
            return None
        combo_path = best_dir / "best_combo.yaml"
        if combo_path.exists():
            return algo_path, combo_path, False
        hp_path = best_dir / "hyperparameter.yaml"
        if hp_path.exists():
            return algo_path, hp_path, True
        return None

    def _find_latest_round_assets(self, model_name: str) -> Optional[Tuple[Path, Path]]:
        base_dir = Path(self.config.output_dir) / self.config.domain / self.config.task_name / model_name
        if not base_dir.exists():
            return None
        candidates: List[Tuple[int, Path]] = []
        for child in base_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("round_"):
                continue
            try:
                round_num = int(child.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            algo_path = child / "algorithm.py"
            hp_path = child / "hyperparameter.yaml"
            if algo_path.exists() and hp_path.exists():
                candidates.append((round_num, child))
        if not candidates:
            return None
        _, latest_dir = max(candidates, key=lambda item: item[0])
        return latest_dir / "algorithm.py", latest_dir / "hyperparameter.yaml"

    def _build_shared_resources(self, model_name: str):
        shared = create_shared_resources(
            domain=self.config.domain,
            task=self.config.task_name,
            algorithms=[model_name],
            nodes=self.config.hpc_nodes,
            gpus_per_node=self.config.hpc_gpus_per_node,
        )

        if not self.config.gpu_ids:
            return shared

        cluster_config = shared.cluster_config
        nodes = self.config.hpc_nodes or (cluster_config.ssh.nodes if cluster_config.ssh else ["localhost"])
        resources = [
            GPUResource(node=node, gpu_id=gpu_id)
            for node in nodes
            for gpu_id in self.config.gpu_ids
        ]
        max_per_gpu = cluster_config.scheduler.max_per_gpu if cluster_config.scheduler else 2
        shared.scheduler = NodeGPUScheduler(resources, max_per_gpu=max_per_gpu)

        if nodes and nodes != ["localhost"]:
            ssh_manager = get_ssh_manager()
            ssh_manager.ensure_connections(nodes)

        return shared

    def _run_round_from_assets(
        self,
        paper_id: str,
        round_num: int,
        algo_path: Path,
        hp_path: Path,
        expand_hyperparameters: bool,
    ) -> RoundResult:
        round_start = time.time()
        self.logger.step(1, 1, "Run PaperBench-Pro from existing outputs")
        model_name = self._get_model_name(paper_id)

        shared = self._build_shared_resources(model_name)
        result_dict = run_with_shared_resources(
            domain=self.config.domain,
            task=self.config.task_name,
            model=model_name,
            algorithm_path=str(algo_path),
            hyperparameter_path=str(hp_path),
            shared=shared,
            timeout=self._paperbench_timeout,
            expand_hyperparameters=expand_hyperparameters,
            output_dir=Path(self.config.output_dir),
        )
        duration = time.time() - round_start
        return RoundResult.from_paperbench(
            data=result_dict,
            round_num=round_num,
            algorithm_path=str(algo_path),
            hyperparameter_path=str(hp_path),
            primary_metric=self.config.target_metric,
            duration=duration,
        )

    def _create_local_graph(self) -> "LocalGraphLoader":
        from graph.local_graph.local_graph_loader import LocalGraphLoader

        return LocalGraphLoader(
            domain=self.config.domain,
            task=self.config.task_name,
            data_root=Path(self.config.data_root),
        )

    def _load_global_graph(self):
        from graph.global_graph.loader import GlobalGraphLoader

        try:
            loader = GlobalGraphLoader(data_root=str(self.config.data_root))
            graph = loader.load(
                domain=self.config.domain,
                task=self.config.task_name,
            )

            if graph is not None:
                self.logger.info(f"GlobalGraph loaded ({len(graph.items)} items)")
            else:
                self.logger.info("GlobalGraph not found or empty")

            return graph
        except Exception as exc:
            self.logger.warning(f"GlobalGraph load failed: {exc}")
            return None

    def _get_model_name(self, paper_id: str) -> str:
        try:
            info = self.local_graph.get_paper_info(paper_id)
            if info and "alias" in info:
                return info["alias"]
        except Exception:
            pass

        parts = paper_id.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return paper_id


def create_agent(
    config_path: str = "ai-scientist/configs/agent.yaml",
    logger_name: Optional[str] = None,
) -> ReplicationAgent:
    return ReplicationAgent(
        config_path=config_path,
        logger_name=logger_name,
    )
