#!/usr/bin/env python3
"""
AI-Scientist 论文复现运行脚本

使用方法:
    # 使用默认配置（复现 BM3_2023）
    python ai-scientist/scripts/run_replication.py
    
    # 指定论文 ID
    python ai-scientist/scripts/run_replication.py --paper_id LATTICE_2021
    
    # 指定配置文件
    python ai-scientist/scripts/run_replication.py --config ai-scientist/configs/config.yaml
    
    # 覆盖配置项
    python ai-scientist/scripts/run_replication.py --task.paper_id=FREEDOM_2023 --agent.max_rounds=5
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# 设置项目根目录
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


def run_batch_replication(
    config,
    paper_ids: List[str],
    llm_config_path: str,
    local_graph,
):
    """
    批量复现多个论文。
    
    每个算法在独立线程中运行自己的迭代循环，共享 GPU 调度器和结果表格。
    算法失败时可立即进入下一轮，无需等待其他算法。
    """
    from agent.replication_agent import ReplicationAgent, ReplicationResult
    from agent.session import SessionStatus, RoundResult
    from llm.client import LLMClient
    from paperbench_pro.interface import create_shared_resources, run_with_shared_resources

    if not paper_ids:
        raise ValueError("No paper IDs provided for batch replication")

    # 1. 初始化所有算法的状态
    states: Dict[str, Dict[str, object]] = {}
    model_names: List[str] = []

    for paper_id in paper_ids:
        llm_client = LLMClient(config_path=llm_config_path)
        logger_name = f"agent_{config.task.domain}_{config.task.task_name}_{paper_id}"
        agent = ReplicationAgent.from_config(
            config=config,
            llm_client=llm_client,
            local_graph=local_graph,
            logger_name=logger_name,
        )
        model_name = agent._get_model_name(paper_id)
        
        if model_name in model_names:
            raise ValueError(f"Duplicate model name detected: {model_name}")
        
        session = agent.session_manager.create_session(
            paper_id=paper_id,
            model_name=model_name,
            domain=config.task.domain,
            task=config.task.task_name,
            max_rounds=config.agent.max_rounds,
        )
        agent.logger.session_start(
            session_id=session.session_id,
            paper_id=paper_id,
            model_name=model_name,
            domain=config.task.domain,
            task=config.task.task_name,
            max_rounds=config.agent.max_rounds,
        )
        session.start()

        states[paper_id] = {
            "agent": agent,
            "session": session,
            "model_name": model_name,
            "current_algo_code": None,
            "current_hp_yaml": None,
            "start_time": time.time(),
            "llm_client": llm_client,
        }
        model_names.append(model_name)

    # 2. 创建共享资源
    shared = create_shared_resources(
        domain=config.task.domain,
        task=config.task.task_name,
        algorithms=model_names,
    )

    # 3. 定义单个算法的完整迭代循环
    def run_single_paper(paper_id: str):
        state = states[paper_id]
        agent = state["agent"]
        session = state["session"]
        model_name = state["model_name"]
        current_algo_code = state["current_algo_code"]
        current_hp_yaml = state["current_hp_yaml"]

        while not session.should_stop():
            round_num = session.current_round
            agent.logger.round_start(round_num, config.agent.max_rounds)
            round_start = time.time()

            # 更新表格显示轮次
            shared.result_table.update_algorithm_round(model_name, round_num)
            shared.renderer.request(clear=True)

            # 生成代码和超参数
            round_result, algo_code, hp_yaml, algo_path, hp_path = agent._generate_round_assets(
                session=session,
                paper_id=paper_id,
                current_algo_code=current_algo_code,
                current_hp_yaml=current_hp_yaml,
            )
            current_algo_code = algo_code
            current_hp_yaml = hp_yaml

            # 如果生成阶段已产生结果（如跳过）
            if round_result is not None:
                session.update_after_round(round_result)
                agent.session_manager.save_session(session)
                agent.logger.round_end(
                    round_num=round_num,
                    success=round_result.error is None,
                    acc=round_result.acc,
                    error=round_result.error,
                    duration=round_result.duration,
                )
                continue

            # 重置表格状态用于新轮次
            shared.result_table.reset_algorithm_for_new_round(model_name)
            shared.renderer.request(clear=True)

            # 运行 paperbench（使用共享资源）
            try:
                pb_result = run_with_shared_resources(
                    domain=config.task.domain,
                    task=config.task.task_name,
                    model=model_name,
                    algorithm_path=str(algo_path),
                    hyperparameter_path=str(hp_path),
                    shared=shared,
                    timeout=config.paperbench_pro.timeout,
                    output_dir=config.paths.output_dir,
                )
                duration = time.time() - round_start
                
                # 直接使用 RoundResult.from_paperbench
                round_result = RoundResult.from_paperbench(
                    data=pb_result,
                    round_num=round_num,
                    algorithm_path=str(algo_path),
                    hyperparameter_path=str(hp_path),
                    primary_metric=agent.config.target_metric,
                    duration=duration,
                )
            except Exception as e:
                duration = time.time() - round_start
                round_result = RoundResult(
                    round=round_num,
                    algorithm_path=str(algo_path),
                    hyperparameter_path=str(hp_path),
                    error=str(e),
                    duration=duration,
                )

            # 更新 session
            session.update_after_round(round_result)
            agent.session_manager.save_session(session)
            agent.logger.round_end(
                round_num=round_num,
                success=round_result.error is None,
                acc=round_result.acc,
                error=round_result.error,
                duration=round_result.duration,
            )

        # 完成
        session.finish()
        finished_time = time.time()
        duration = finished_time - state["start_time"]
        agent.logger.session_end(
            success=session.status == SessionStatus.SUCCESS,
            best_acc=session.best_acc,
            best_round=session.best_round,
            total_rounds=session.current_round,
            duration=duration,
            total_tokens=0,
            output_dir=session.output_dir,
        )
        state["llm_client"].close()
        state["finished_time"] = finished_time

        # 保存最佳结果
        if session.status == SessionStatus.SUCCESS:
            agent._save_best_result(session)

    # 4. 并行运行所有算法
    with ThreadPoolExecutor(max_workers=len(paper_ids)) as executor:
        futures = {executor.submit(run_single_paper, pid): pid for pid in paper_ids}
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] {paper_id} failed: {e}")

    # 5. 收集结果
    results: List[ReplicationResult] = []
    for paper_id in paper_ids:
        state = states[paper_id]
        session = state["session"]
        finished_time = state.get("finished_time")
        duration = (finished_time - state["start_time"]) if finished_time else (time.time() - state["start_time"])
        results.append(
            ReplicationResult(
                session_id=session.session_id,
                paper_id=paper_id,
                model_name=state["model_name"],
                success=session.status == SessionStatus.SUCCESS,
                best_acc=session.best_acc,
                best_round=session.best_round,
                total_rounds=session.current_round,
                final_status=session.status.value,
                output_dir=session.output_dir,
                duration=duration,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="AI-Scientist 论文复现")
    parser.add_argument(
        "--config",
        type=str,
        default="ai-scientist/configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--paper_id",
        type=str,
        default=None,
        help="目标论文 ID（覆盖配置文件中的设置）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印配置，不实际运行",
    )
    parser.add_argument(
        "--use_latest_outputs",
        action="store_true",
        help="跳过生成，直接运行最近一次输出的算法与超参",
    )
    
    args, unknown_args = parser.parse_known_args()
    
    # 加载配置
    from common.config import load_config
    
    config = load_config(
        config_path=args.config,
        cli_args=unknown_args,
    )
    
    # 如果指定了 paper_id，覆盖配置
    paper_override = _parse_paper_id_arg(args.paper_id)
    if paper_override is not None:
        config.task.paper_id = paper_override

    paper_ids = _normalize_paper_ids(config.task)
    
    print("=" * 60)
    print("  AI-Scientist 论文复现")
    print("=" * 60)
    print(f"  Domain:     {config.task.domain}")
    print(f"  Task:       {config.task.task_name}")
    paper_display = ", ".join(paper_ids) if paper_ids else "N/A"
    print(f"  Paper ID:   {paper_display}")
    print(f"  Max Rounds: {config.agent.max_rounds}")
    print(f"  Output:     {config.paths.output_dir}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[Dry Run] 配置加载成功，不实际运行")
        return 0
    
    # 初始化 LLM 客户端
    from llm.client import LLMClient
    
    llm_config_path = str(PROJECT_ROOT / "ai-scientist" / "configs" / "llm.yaml")
    print(f"\n正在初始化 LLM 客户端...")
    llm_client = LLMClient(config_path=llm_config_path)
    print(f"  Model: {llm_client.model_name}")
    print(f"  Provider: {llm_client.provider}")
    
    # 初始化 LocalGraphLoader
    from graph.local_graph.local_graph_loader import LocalGraphLoader
    
    print(f"\n正在加载知识图谱...")
    local_graph = LocalGraphLoader(
        domain=config.task.domain,
        task=config.task.task_name,
        data_root=config.paths.data_root,
    )
    print(f"  Papers: {len(local_graph.list_papers())}")
    
    # 运行复现
    print(f"\n{'=' * 60}")
    print(f"  开始复现: {paper_display}")
    print(f"{'=' * 60}\n")

    if len(paper_ids) <= 1:
        # 单论文模式
        from agent.replication_agent import ReplicationAgent
        
        print(f"\n正在创建 Agent...")
        agent = ReplicationAgent.from_config(
            config=config,
            llm_client=llm_client,
            local_graph=local_graph,
        )
        
        paper_id = paper_ids[0] if paper_ids else config.task.paper_id
        result = agent.run(
            paper_id=paper_id,
        )
        
        # 输出结果
        print(f"\n{'=' * 60}")
        print(f"  复现完成")
        print(f"{'=' * 60}")
        print(f"  Session ID:   {result.session_id}")
        print(f"  Success:      {result.success}")
        print(f"  Best Acc:     {result.best_acc:.4f}" if result.best_acc else "  Best Acc:     N/A")
        print(f"  Best Round:   {result.best_round}")
        print(f"  Total Rounds: {result.total_rounds}")
        print(f"  Duration:     {result.duration:.1f}s")
        print(f"  Output:       {result.output_dir}")
        print(f"{'=' * 60}")
        
        llm_client.close()
        return 0 if result.success else 1

    # 多论文批量模式
    print(f"\n正在创建 Batch Agent...")
    if args.use_latest_outputs:
        print("\n[Error] --use_latest_outputs 仅支持单论文模式")
        llm_client.close()
        return 1

    results = run_batch_replication(
        config=config,
        paper_ids=paper_ids,
        llm_config_path=llm_config_path,
        local_graph=local_graph,
    )
    
    print(f"\n{'=' * 60}")
    print(f"  批量复现完成")
    print(f"{'=' * 60}")
    for result in results:
        best_acc = f"{result.best_acc:.4f}" if result.best_acc else "N/A"
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {result.paper_id} ({result.model_name})  {status}  best={best_acc}  rounds={result.total_rounds}")
    print(f"{'=' * 60}")
    
    llm_client.close()
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
