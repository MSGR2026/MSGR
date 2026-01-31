from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from common.types import Content
from storage.json_storage import read_json, slugify_component, write_json_atomic


class ContentStore:
    """
    Local graph Content storage (file-backed JSON).

    Layout (requested):
      data_root/graphs/local/content/{paper_id}.json
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.content_dir = self.data_root / "graphs" / "local" / "content"

    def _content_path(self, paper_id: str) -> Path:
        return self.content_dir / f"{slugify_component(paper_id)}.json"

    def save(self, content: Content) -> None:
        path = self._content_path(content.paper_id)
        write_json_atomic(path, content.to_dict())

    def get(self, paper_id: str) -> Optional[Content]:
        path = self._content_path(paper_id)
        data = read_json(path, default=None)
        if isinstance(data, dict):
            return Content.from_dict(data)
        return None

    def list(self, domain: str | None = None, task: str | None = None) -> List[Content]:
        out: List[Content] = []
        if not self.content_dir.exists():
            return out
        for p in sorted(self.content_dir.glob("*.json")):
            data = read_json(p, default=None)
            if not isinstance(data, dict):
                continue
            c = Content.from_dict(data)
            if domain is not None and c.domain != domain:
                continue
            if task is not None and task not in c.task:
                continue
            out.append(c)
        return out

    def delete(self, paper_id: str) -> bool:
        path = self._content_path(paper_id)
        if path.exists():
            path.unlink()
            return True
        return False

