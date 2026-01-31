from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from common.types import Implementation, datetime_to_iso, now_utc
from storage.json_storage import load_index, read_json, save_index, slugify_component, write_json_atomic


class ImplementationStore:
    """
    Local graph Implementation storage (file-backed JSON) + impl_index maintenance.
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.impl_dir = self.data_root / "graphs" / "local" / "implementations"
        self.gt_dir = self.impl_dir / "ground_truth"
        self.generated_dir = self.impl_dir / "generated"
        self.index_dir = self.data_root / "index"
        self.impl_index_file = self.index_dir / "impl_index.json"

    def _load_impl_index(self) -> Dict[str, Any]:
        return load_index(self.impl_index_file, root_key="implementations")

    def _save_impl_index(self, data: Dict[str, Any]) -> None:
        data["updated_at"] = datetime_to_iso(now_utc())
        save_index(self.impl_index_file, data)

    def _ground_truth_path(self, paper_id: str) -> Path:
        return self.gt_dir / f"{slugify_component(paper_id)}_gt.json"

    def _generated_path(self, session_id: str, round_num: int) -> Path:
        sess_dir = self.generated_dir / slugify_component(session_id)
        return sess_dir / f"round_{int(round_num)}.json"

    def save(self, impl: Implementation) -> None:
        if impl.source == "ground_truth":
            path = self._ground_truth_path(impl.paper_id)
        else:
            if not impl.session_id:
                raise ValueError("agent_generated implementation must provide session_id")
            path = self._generated_path(impl.session_id, impl.round)
            # Avoid accidental overwrite if the same round file already stores a different impl.
            if path.exists():
                existing = read_json(path, default=None)
                if isinstance(existing, dict) and existing.get("impl_id") not in {None, impl.impl_id}:
                    path = path.with_name(f"round_{int(impl.round)}_{slugify_component(impl.paper_id)}.json")

        write_json_atomic(path, impl.to_dict())

        idx = self._load_impl_index()
        idx["implementations"][impl.impl_id] = {
            "paper_id": impl.paper_id,
            "source": impl.source,
            "path": str(path.relative_to(self.data_root)),
            "session_id": impl.session_id,
            "round": int(impl.round),
            "acc": impl.acc,
            "updated_at": datetime_to_iso(now_utc()),
        }
        self._save_impl_index(idx)

    def get(self, impl_id: str) -> Optional[Implementation]:
        idx = self._load_impl_index()
        meta = idx["implementations"].get(impl_id)
        if isinstance(meta, dict) and meta.get("path"):
            path = self.data_root / str(meta["path"])
            data = read_json(path, default=None)
            if isinstance(data, dict):
                return Implementation.from_dict(data)
        return None

    def get_ground_truth(self, paper_id: str) -> Optional[Implementation]:
        path = self._ground_truth_path(paper_id)
        data = read_json(path, default=None)
        if isinstance(data, dict):
            return Implementation.from_dict(data)
        return None

    def list(self, paper_id: str | None = None, session_id: str | None = None) -> List[Implementation]:
        idx = self._load_impl_index()
        out: List[Implementation] = []
        for impl_id, meta in idx["implementations"].items():
            if not isinstance(meta, dict):
                continue
            if paper_id is not None and meta.get("paper_id") != paper_id:
                continue
            if session_id is not None and meta.get("session_id") != session_id:
                continue
            impl = self.get(impl_id)
            if impl is not None:
                out.append(impl)
        return out

