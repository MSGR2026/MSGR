from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def slugify_component(value: str) -> str:
    """
    Convert an arbitrary string into a filesystem-friendly path component.
    Keeps letters/numbers and '._-' characters, replaces others with '_'.
    """
    v = (value or "").strip()
    if not v:
        return "unknown"
    v = re.sub(r"[^A-Za-z0-9._-]+", "_", v)
    v = re.sub(r"_+", "_", v).strip("_")
    return v or "unknown"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Be robust to manually edited / partially written files.
        return default


def write_json_atomic(path: Path, data: Any, *, indent: int = 2) -> None:
    """
    Atomic JSON writer: write to a temp file in the same directory and replace.
    """
    ensure_parent_dir(path)
    tmp_dir = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(tmp_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=False)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            # Best-effort cleanup; file is either replaced or safe to ignore.
            pass


def load_index(path: Path, *, root_key: str) -> Dict[str, Any]:
    data = read_json(path, default=None)
    if not isinstance(data, dict):
        return {root_key: {}}
    if root_key not in data or not isinstance(data.get(root_key), dict):
        data[root_key] = {}
    return data


def save_index(path: Path, data: Dict[str, Any]) -> None:
    write_json_atomic(path, data, indent=2)


