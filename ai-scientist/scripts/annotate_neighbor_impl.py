#!/usr/bin/env python3
"""
Annotate neighbor implementations with per-target comments.

Writes annotated neighbor code to:
  data/{domain}/{task}/local/implementation/{TARGET_ID}/neighbors/{NEIGHBOR_ID}/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SRC_ROOT = REPO_ROOT / "ai-scientist" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.config import get_repo_root, load_yaml, resolve_data_root
from graph.local_graph.local_graph_loader import LocalGraphLoader
from llm.client import LLMClient
from storage.domain_task_storage import DomainTaskStorage
from storage.json_storage import slugify_component


ALG_MARK = "===ALGORITHM==="
HP_MARK = "===HYPERPARAMETER==="


def _load_data_root() -> Path:
    repo_root = get_repo_root()
    config_path = repo_root / "ai-scientist" / "configs" / "storage.yaml"
    if config_path.exists():
        cfg = load_yaml(config_path)
        return resolve_data_root(cfg.get("data_root"))
    default_root = resolve_data_root()
    alt_root = repo_root / "ai-scientist" / "data"
    if alt_root.exists():
        return alt_root
    return default_root


def _iter_neighbors(
    loader: LocalGraphLoader,
    target_id: str,
    source_id: Optional[str],
) -> List[str]:
    if source_id:
        return [source_id]
    neighbors = loader.get_neighbors(target_id, k=1, edge_type="in-domain")
    return sorted({n.paper_id for n in neighbors if n.paper_id})


def _build_prompt(
    target_id: str,
    neighbor_id: str,
    target_algorithm: str,
    target_hyperparameter: str,
    neighbor_algorithm: str,
    neighbor_hyperparameter: str,
    comment_mode: str,
    max_comments: int,
) -> tuple[str, str]:
    mode_note = ""
    if comment_mode == "balanced":
        mode_note = (
            f"Add only key REUSE and key DIFFERENT comments. "
            f"Limit to at most {max_comments} comments per file. "
            "Prefer block-level comments near core modules or steps. "
            "Skip minor similarities or repeated notes."
        )
    elif comment_mode == "diff-only":
        mode_note = (
            f"Only add DIFFERENT comments (no SIMILAR/REUSE). "
            f"Limit to at most {max_comments} comments per file. "
            "Prefer block-level comments near the changed sections. "
            "If only minor differences exist, add a single summary comment at top."
        )
    else:
        mode_note = "You may annotate more densely when needed."

    system_prompt = (
        "You are an expert ML engineer. "
        "Add inline comments to the NEIGHBOR implementation only using str_replace edits. "
        "Do not change code or config values beyond inserting comments. "
        "Use only '#' comments with one of these prefixes: "
        "'# SIMILAR:', '# REUSE:', '# DIFFERENT:'. "
        "Comments should explain similarity, direct reuse, or differences "
        "relative to the TARGET implementation. "
        f"{mode_note}"
    )
    user_prompt = (
        f"TARGET_ID: {target_id}\n"
        f"NEIGHBOR_ID: {neighbor_id}\n\n"
        "Return str_replace edits for NEIGHBOR algorithm and hyperparameters.\n"
        "Use XML tool_call blocks in each section, for example:\n"
        "<tool_call>\n<name>str_replace</name>\n"
        "<old_str>...</old_str>\n<new_str>...</new_str>\n</tool_call>\n"
        "You may include multiple tool_call blocks per section.\n"
        "Output format (no code fences, no extra text):\n"
        f"{ALG_MARK}\n"
        "<str_replace tool_call blocks for neighbor python code>\n"
        f"{HP_MARK}\n"
        "<str_replace tool_call blocks for neighbor yaml>\n\n"
        "TARGET algorithm (python):\n"
        f"{target_algorithm}\n\n"
        "TARGET hyperparameter (yaml):\n"
        f"{target_hyperparameter}\n\n"
        "NEIGHBOR algorithm (python):\n"
        f"{neighbor_algorithm}\n\n"
        "NEIGHBOR hyperparameter (yaml):\n"
        f"{neighbor_hyperparameter}\n"
    )
    return system_prompt, user_prompt


def _clean_block(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_blocks(text: str) -> tuple[str, str]:
    start_alg = text.find(ALG_MARK)
    start_hp = text.find(HP_MARK)
    if start_alg == -1 or start_hp == -1:
        raise ValueError("LLM output missing required markers.")
    if start_alg > start_hp:
        raise ValueError("LLM output markers are out of order.")
    alg_body = text[start_alg + len(ALG_MARK):start_hp]
    hp_body = text[start_hp + len(HP_MARK):]
    alg_body = _clean_block(alg_body)
    hp_body = _clean_block(hp_body)
    return alg_body, hp_body


def _apply_str_replace_xml(response: str, current_content: str) -> str:
    pattern = (
        r"<tool_call>\s*<name>str_replace</name>\s*"
        r"<old_str>\s*(.*?)\s*</old_str>\s*"
        r"<new_str>\s*(.*?)\s*</new_str>\s*</tool_call>"
    )
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        print("[warn] no valid str_replace tool_call found; keeping current content")
        return current_content

    updated_content = current_content
    for i, (old_str, new_str) in enumerate(matches):
        old_str = old_str.strip("\n")
        new_str = new_str.strip("\n")
        if old_str in updated_content:
            updated_content = updated_content.replace(old_str, new_str, 1)
            print(f"[info] str_replace[{i + 1}]: {len(old_str)} chars -> {len(new_str)} chars")
        else:
            print(f"[warn] str_replace[{i + 1}] old_str not found")
    return updated_content


def _apply_str_replace(response: str, current_content: str) -> str:
    pattern = r"```str_replace\s*\n<<<old>>>\s*\n(.*?)<<<new>>>\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        pattern2 = r"<<<old>>>\s*\n(.*?)<<<new>>>\s*\n(.*?)(?=<<<|$)"
        matches = re.findall(pattern2, response, re.DOTALL)

    if not matches:
        print("[warn] no valid str_replace block found; keeping current content")
        return current_content

    updated_content = current_content
    for i, (old_str, new_str) in enumerate(matches):
        old_str = old_str.strip()
        new_str = new_str.strip()
        if old_str in updated_content:
            updated_content = updated_content.replace(old_str, new_str, 1)
            print(f"[info] str_replace[{i + 1}]: {len(old_str)} chars -> {len(new_str)} chars")
        else:
            print(f"[warn] str_replace[{i + 1}] old_str not found")
    return updated_content


def _process_str_replace_block(
    block_text: str,
    current_content: str,
    label: str,
) -> str:
    skip_pattern = rf"<tool_call>\s*<name>skip_{label}</name>\s*</tool_call>"
    if re.search(skip_pattern, block_text):
        print(f"[info] skip_{label} received; keeping current content")
        return current_content

    if "<tool_call>" in block_text and "<name>str_replace</name>" in block_text:
        return _apply_str_replace_xml(block_text, current_content)

    if "<<<old>>>" in block_text and "<<<new>>>" in block_text:
        return _apply_str_replace(block_text, current_content)

    if block_text.strip():
        print(f"[warn] no str_replace markers for {label}; using full replacement")
        return block_text.strip()

    print(f"[warn] empty {label} block; keeping current content")
    return current_content


def _write_neighbor_override(
    storage: DomainTaskStorage,
    domain: str,
    task: str,
    target_id: str,
    neighbor_id: str,
    algorithm: str,
    hyperparameter: str,
) -> Path:
    impl_dir = storage._implementation_dir(domain, task)
    neighbor_dir = (
        impl_dir
        / slugify_component(target_id)
        / "neighbors"
        / slugify_component(neighbor_id)
    )
    neighbor_dir.mkdir(parents=True, exist_ok=True)
    algo_path = neighbor_dir / "algorithm.py"
    hp_path = neighbor_dir / "hyperparameter.yaml"
    algo_path.write_text(algorithm.rstrip() + "\n", encoding="utf-8")
    hp_path.write_text(hyperparameter.rstrip() + "\n", encoding="utf-8")
    return neighbor_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Annotate neighbor implementations with target-specific comments.",
    )
    parser.add_argument("--domain", required=True, help="Domain name, e.g., Recsys")
    parser.add_argument("--task", required=True, help="Task name, e.g., ContextAware")
    parser.add_argument("--target_id", required=True, help="Target paper ID")
    parser.add_argument(
        "--source_id",
        default=None,
        help="Optional neighbor paper ID; if omitted, process all in-domain neighbors",
    )
    parser.add_argument(
        "--llm_config",
        default="ai-scientist/configs/llm.yaml",
        help="Path to LLM config YAML",
    )
    parser.add_argument(
        "--comment_mode",
        choices=["balanced", "sparse", "diff-only", "key-reuse", "dense"],
        default="balanced",
        help="Comment density mode.",
    )
    parser.add_argument(
        "--max_comments",
        type=int,
        default=10,
        help="Max comments per file for sparse/diff-only modes.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    data_root = _load_data_root()
    storage = DomainTaskStorage(data_root)
    loader = LocalGraphLoader(domain=args.domain, task=args.task, data_root=data_root)

    target_impl = storage.get_implementation(args.domain, args.task, args.target_id)
    if target_impl is None:
        print(f"[error] target implementation not found: {args.target_id}")
        return 1

    neighbors = _iter_neighbors(loader, args.target_id, args.source_id)
    if not neighbors:
        print("[info] no neighbors to process")
        return 0

    repo_root = get_repo_root()
    llm_config = args.llm_config
    if not Path(llm_config).is_absolute():
        llm_config = str(repo_root / llm_config)

    client = LLMClient(config_path=llm_config)
    try:
        for neighbor_id in neighbors:
            neighbor_impl = storage.get_implementation(args.domain, args.task, neighbor_id)
            if neighbor_impl is None:
                print(f"[warn] neighbor implementation missing: {neighbor_id}")
                continue

            system_prompt, user_prompt = _build_prompt(
                target_id=args.target_id,
                neighbor_id=neighbor_id,
                target_algorithm=target_impl.algorithm,
                target_hyperparameter=target_impl.hyperparameter,
                neighbor_algorithm=neighbor_impl.algorithm,
                neighbor_hyperparameter=neighbor_impl.hyperparameter,
                comment_mode=args.comment_mode,
                max_comments=args.max_comments,
            )
            response = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            try:
                alg_text, hp_text = _extract_blocks(response)
            except ValueError as exc:
                print(f"[error] failed to parse LLM output for {neighbor_id}: {exc}")
                continue

            updated_algorithm = _process_str_replace_block(
                alg_text,
                neighbor_impl.algorithm,
                "algorithm",
            )
            updated_hyperparameter = _process_str_replace_block(
                hp_text,
                neighbor_impl.hyperparameter,
                "hyperparameter",
            )

            out_dir = _write_neighbor_override(
                storage=storage,
                domain=args.domain,
                task=args.task,
                target_id=args.target_id,
                neighbor_id=neighbor_id,
                algorithm=updated_algorithm,
                hyperparameter=updated_hyperparameter,
            )
            print(f"[ok] wrote neighbor override: {out_dir}")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
