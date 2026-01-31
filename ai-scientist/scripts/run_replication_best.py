#!/usr/bin/env python3
"""
Best-only replication runner.

Loads the best available outputs and runs PaperBenchPro without LLM generation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai-scientist" / "src"))


def _parse_paper_id_arg(paper_id_arg: Optional[str]) -> Optional[object]:
    if not paper_id_arg:
        return None
    if "," in paper_id_arg:
        return [p.strip() for p in paper_id_arg.split(",") if p.strip()]
    return paper_id_arg.strip()


def _normalize_paper_ids(task_config) -> List[str]:
    if hasattr(task_config, "paper_ids"):
        return task_config.paper_ids()
    paper_id = getattr(task_config, "paper_id", "")
    if isinstance(paper_id, list):
        return [p for p in paper_id if p]
    if isinstance(paper_id, str) and paper_id.strip():
        return [paper_id.strip()]
    return []


def _parse_csv_ints(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_strings(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-Scientist replication (best-only)")
    parser.add_argument(
        "--config",
        type=str,
        default="ai-scientist/configs/config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--paper_id",
        type=str,
        default=None,
        help="Paper ID override (comma-separated for multiple)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print config, do not run",
    )
    parser.add_argument(
        "--use_latest_outputs",
        action="store_true",
        help="Skip best/ and use latest round outputs",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="GPU ids to use (comma-separated, e.g. 0,1)",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Override nodes list (comma-separated)",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=None,
        help="Override GPUs per node",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=None,
        help="Override max concurrent tasks",
    )

    args, unknown_args = parser.parse_known_args()

    from common.config import load_config

    config = load_config(
        config_path=args.config,
        cli_args=unknown_args,
    )

    paper_override = _parse_paper_id_arg(args.paper_id)
    if paper_override is not None:
        config.task.paper_id = paper_override

    if args.gpu_ids is not None:
        config.resources.gpu_ids = _parse_csv_ints(args.gpu_ids)

    if args.nodes or args.gpus_per_node is not None or args.max_concurrent is not None:
        config._raw.setdefault("execution", {})
        if args.nodes:
            config._raw["execution"]["nodes"] = _parse_csv_strings(args.nodes)
        if args.gpus_per_node is not None:
            config._raw["execution"]["gpus_per_node"] = args.gpus_per_node
        if args.max_concurrent is not None:
            config._raw["execution"]["max_concurrent"] = args.max_concurrent

    paper_ids = _normalize_paper_ids(config.task)

    print("=" * 60)
    print("  AI-Scientist Replication (Best-Only)")
    print("=" * 60)
    print(f"  Domain:     {config.task.domain}")
    print(f"  Task:       {config.task.task_name}")
    paper_display = ", ".join(paper_ids) if paper_ids else "N/A"
    print(f"  Paper ID:   {paper_display}")
    print(f"  Output:     {config.paths.output_dir}")
    if config.resources.gpu_ids:
        print(f"  GPU IDs:    {config.resources.gpu_ids}")
    exec_cfg = config._raw.get("execution", {})
    if exec_cfg:
        print(f"  Nodes:      {exec_cfg.get('nodes', 'default')}")
        print(f"  GPUs/Node:  {exec_cfg.get('gpus_per_node', 'default')}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry Run] Config loaded, no execution.")
        return 0

    from graph.local_graph.local_graph_loader import LocalGraphLoader
    from agent.replication_agent_best import ReplicationAgent

    print("\nLoading local graph...")
    local_graph = LocalGraphLoader(
        domain=config.task.domain,
        task=config.task.task_name,
        data_root=config.paths.data_root,
    )
    print(f"  Papers: {len(local_graph.list_papers())}")

    results = []
    for paper_id in (paper_ids or [config.task.paper_id]):
        logger_name = f"agent_best_{config.task.domain}_{config.task.task_name}_{paper_id}"
        agent = ReplicationAgent.from_config(
            config=config,
            local_graph=local_graph,
            logger_name=logger_name,
        )
        print("\n" + "=" * 60)
        print(f"  Running: {paper_id}")
        print("=" * 60)
        start = time.time()
        result = agent.run(
            paper_id=paper_id,
            prefer_best_outputs=not args.use_latest_outputs,
            fallback_to_latest=True,
        )
        result.duration = max(result.duration, time.time() - start)
        results.append(result)

    print("\n" + "=" * 60)
    print("  Replication Finished")
    print("=" * 60)
    for result in results:
        best_acc = f"{result.best_acc:.4f}" if result.best_acc else "N/A"
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {result.paper_id} ({result.model_name})  {status}  best={best_acc}  rounds={result.total_rounds}")
    print("=" * 60)

    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
