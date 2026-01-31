from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from common.config import get_repo_root
from common.types import Content, Implementation, now_utc
from storage.interface import GraphStorageInterface


@dataclass(frozen=True)
class MmrecMetricRecord:
    model: str
    dataset: str
    returncode: int
    metrics: Dict[str, Any]
    log_file: str = ""
    config: str = ""


def _mmrec_root(repo_root: Path) -> Path:
    return repo_root / "paperbench_pro" / "Recsys" / "MMRec"


def _mmrec_model_paths(repo_root: Path, model_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Resolve (algorithm_py, hyperparameter_yaml) for a MMRec model name.
    """
    mmrec = _mmrec_root(repo_root)
    model_py = mmrec / "src" / "models" / f"{model_name.lower()}.py"
    cfg_yaml = mmrec / "src" / "configs" / "model" / f"{model_name}.yaml"

    if not model_py.exists():
        # Case-insensitive fallback
        candidates = list((mmrec / "src" / "models").glob("*.py"))
        for c in candidates:
            if c.stem.lower() == model_name.lower():
                model_py = c
                break
    if not model_py.exists():
        model_py = None

    if not cfg_yaml.exists():
        candidates = list((mmrec / "src" / "configs" / "model").glob("*.yaml"))
        for c in candidates:
            if c.stem.lower() == model_name.lower():
                cfg_yaml = c
                break
    if not cfg_yaml.exists():
        cfg_yaml = None

    return model_py, cfg_yaml


def _metrics_files_for_dataset(repo_root: Path, dataset: str) -> List[Path]:
    mmrec = _mmrec_root(repo_root)
    # Repo contains: model_yaml_metrics.jsonl (baby), model_yaml_metrics_clothing.jsonl, model_yaml_metrics_sports.jsonl
    patterns = [
        mmrec / "model_yaml_metrics.jsonl",
        mmrec / "model_yaml_metrics_clothing.jsonl",
        mmrec / "model_yaml_metrics_sports.jsonl",
    ]
    files = [p for p in patterns if p.exists()]
    # We'll filter by `dataset` inside lines anyway.
    return files


def load_mmrec_metrics(repo_root: Path, dataset: str) -> Dict[str, MmrecMetricRecord]:
    """
    Load per-model metrics for a given dataset from MMRec jsonl logs.
    Returns mapping: model_name -> record
    """
    out: Dict[str, MmrecMetricRecord] = {}
    for p in _metrics_files_for_dataset(repo_root, dataset):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("dataset") != dataset:
                continue
            model = obj.get("model")
            if not model:
                continue
            out[str(model)] = MmrecMetricRecord(
                model=str(model),
                dataset=str(dataset),
                returncode=int(obj.get("returncode") or 0),
                metrics=dict(obj.get("metrics") or {}),
                log_file=str(obj.get("log_file") or ""),
                config=str(obj.get("config") or ""),
            )
    return out


def _default_model_name_for_content(c: Content) -> Optional[str]:
    if c.model_name:
        return c.model_name
    # Fallback: paper_id like "BM3_2023" -> "BM3"
    if "_" in c.paper_id:
        return c.paper_id.split("_", 1)[0]
    return None


def _load_graph_out_code_map(graph_out_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Load (model_code, config_yaml) from graph_out.json nodes by paper id.
    """
    if not graph_out_path.exists():
        return {}
    try:
        doc = json.loads(graph_out_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    nodes = doc.get("nodes", [])
    if not isinstance(nodes, list):
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        pid = n.get("id") or n.get("paper_id")
        if not pid:
            continue
        model_code = n.get("model_code") or ""
        config_yaml = n.get("config_yaml") or ""
        if isinstance(model_code, str) and isinstance(config_yaml, str) and (model_code or config_yaml):
            out[str(pid)] = (model_code, config_yaml)
    return out


def import_mmrec_ground_truth(
    storage: GraphStorageInterface,
    *,
    dataset: str = "baby",
    only_models: Optional[Set[str]] = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    For each stored Content, if it maps to a MMRec model, create/save a ground_truth Implementation:
      - algorithm_path: paperbench_pro/Recsys/MMRec/src/models/<model>.py
      - hyperparameter_path: paperbench_pro/Recsys/MMRec/src/configs/model/<MODEL>.yaml
      - metrics: from MMRec model_yaml_metrics*.jsonl (if available)

    Returns stats dict.
    """
    repo_root = get_repo_root()
    metrics_by_model = load_mmrec_metrics(repo_root, dataset=dataset)

    # Prefer algorithm/config content from data_root/graph_out.json if present (aligns with graph_out.json).
    graph_out_path = None
    try:
        data_root = getattr(storage, "data_root", None)
        if data_root is not None:
            graph_out_path = Path(data_root) / "graph_out.json"
    except Exception:
        graph_out_path = None
    graph_out_code_map = _load_graph_out_code_map(graph_out_path) if graph_out_path is not None else {}

    imported = 0
    skipped = 0
    missing_paths = 0

    for c in storage.list_contents():
        model_name = _default_model_name_for_content(c)
        if not model_name:
            skipped += 1
            continue
        if only_models is not None and model_name not in only_models:
            skipped += 1
            continue

        impl_id = f"{c.paper_id}_gt"
        if not overwrite and storage.get_implementation(impl_id) is not None:
            skipped += 1
            continue

        algo_path, hp_path = _mmrec_model_paths(repo_root, model_name=model_name)
        if algo_path is None or hp_path is None:
            missing_paths += 1
            continue

        # Store repo-relative paths for portability
        try:
            algo_str = str(algo_path.relative_to(repo_root))
        except Exception:
            algo_str = str(algo_path)
        try:
            hp_str = str(hp_path.relative_to(repo_root))
        except Exception:
            hp_str = str(hp_path)

        # Inline contents (preferred)
        algo_text = ""
        hp_text = ""
        if c.paper_id in graph_out_code_map:
            algo_text, hp_text = graph_out_code_map[c.paper_id]
        if not algo_text:
            algo_text = algo_path.read_text(encoding="utf-8", errors="replace")
        if not hp_text:
            hp_text = hp_path.read_text(encoding="utf-8", errors="replace")

        rec = metrics_by_model.get(model_name)
        metrics: Dict[str, Any] = {}
        acc: Optional[float] = None
        error_msg: Optional[str] = None
        if rec is not None:
            metrics = dict(rec.metrics)
            metrics["dataset"] = dataset
            metrics["mmrec_returncode"] = rec.returncode
            if rec.log_file:
                metrics["mmrec_log_file"] = rec.log_file
            if rec.config:
                metrics["mmrec_config_from_log"] = rec.config
            # Choose a primary scalar for `acc`
            if "recall@20" in metrics:
                acc = float(metrics["recall@20"])
            elif "map@20" in metrics:
                acc = float(metrics["map@20"])
            if rec.returncode != 0:
                error_msg = f"MMRec run returned non-zero returncode={rec.returncode}"

        impl = Implementation(
            impl_id=impl_id,
            paper_id=c.paper_id,
            algorithm=algo_text,
            hyperparameter=hp_text,
            metrics=metrics,
            algorithm_path=algo_str,
            hyperparameter_path=hp_str,
            acc=acc,
            error_msg=error_msg,
            source="ground_truth",
            session_id=None,
            round=0,
            created_at=now_utc(),
        )
        storage.save_implementation(impl)
        imported += 1

    return {"imported": imported, "skipped": skipped, "missing_paths": missing_paths}


