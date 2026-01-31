from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config import get_repo_root, load_yaml, resolve_data_root
from common.types import Content, GlobalGraphItem, Implementation, SemanticEdge, datetime_to_iso, now_utc
from storage.content_store import ContentStore
from storage.edge_store import EdgeStore
from storage.implementation_store import ImplementationStore
from storage.interface import GraphStorageInterface
from storage.json_storage import read_json, slugify_component, write_json_atomic


class GraphRepository(GraphStorageInterface):
    """
    JSON-file-backed repository implementing GraphStorageInterface.

    All data is stored under:
      <data_root>/graphs/...
      <data_root>/index/...
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        # Keep the raw local graph file as an optional artifact, but storage uses
        # graphs/local/{content,edge,implementations} as requested.
        self.graph_out_path = self.data_root / "graph_out.json"

        self.content_store = ContentStore(data_root=data_root)
        self.edge_store = EdgeStore(data_root=data_root)
        self.impl_store = ImplementationStore(data_root=data_root)

        self.global_dir = self.data_root / "graphs" / "global"

    @classmethod
    def from_default_config(cls) -> "GraphRepository":
        repo_root = get_repo_root()
        config_path = repo_root / "ai-scientist" / "configs" / "storage.yaml"
        cfg = load_yaml(config_path)
        data_root = resolve_data_root(cfg.get("data_root"))
        return cls(data_root=data_root)

    # ============ Content ============
    def get_content(self, paper_id: str) -> Optional[Content]:
        return self.content_store.get(paper_id)

    def save_content(self, content: Content) -> None:
        self.content_store.save(content)

    def list_contents(self, domain: str | None = None, task: str | None = None) -> List[Content]:
        return self.content_store.list(domain=domain, task=task)

    def delete_content(self, paper_id: str) -> bool:
        return self.content_store.delete(paper_id)

    # ============ Implementation ============
    def get_implementation(self, impl_id: str) -> Optional[Implementation]:
        return self.impl_store.get(impl_id)

    def get_ground_truth(self, paper_id: str) -> Optional[Implementation]:
        return self.impl_store.get_ground_truth(paper_id)

    def save_implementation(self, impl: Implementation) -> None:
        self.impl_store.save(impl)

    def list_implementations(self, paper_id: str | None = None, session_id: str | None = None) -> List[Implementation]:
        return self.impl_store.list(paper_id=paper_id, session_id=session_id)

    # ============ Edge ============
    def _domain_for_paper(self, paper_id: str) -> Optional[str]:
        c = self.get_content(paper_id)
        return c.domain if c else None

    def save_edge(self, edge: SemanticEdge) -> None:
        self.edge_store.upsert_edge(edge)

    def get_edges(self, domain: str) -> List[SemanticEdge]:
        # Edges are stored in one file; filter by domain inferred from Content.
        out: List[SemanticEdge] = []
        for e in self.edge_store.list_edges():
            src = self.get_content(e.src_paper_id)
            dst = self.get_content(e.dst_paper_id)
            if (src and src.domain == domain) or (dst and dst.domain == domain):
                out.append(e)
        return out

    def get_neighbors(self, paper_id: str, k: int = 5, edge_type: str | None = None) -> List[SemanticEdge]:
        candidates = []
        for e in self.edge_store.list_edges():
            if e.src_paper_id != paper_id:
                continue
            if edge_type is not None and e.edge_type != edge_type:
                continue
            candidates.append(e)

        candidates.sort(key=lambda x: x.weight, reverse=True)
        return candidates[: max(0, int(k))]

    # ============ Global Graph ============
    def _global_domain_dir(self, domain: str) -> Path:
        return self.global_dir / slugify_component(domain)

    def _task_slug(self, task: str) -> str:
        # Task might include spaces or slashes; normalize for filename.
        return slugify_component(task)

    def list_global_graph_versions(self, domain: str, task: str) -> List[int]:
        d = self._global_domain_dir(domain)
        if not d.exists():
            return []
        task_slug = self._task_slug(task)
        versions: List[int] = []
        for p in d.glob(f"{task_slug}_v*.json"):
            m = re.match(rf"^{re.escape(task_slug)}_v(\d+)\.json$", p.name)
            if m:
                versions.append(int(m.group(1)))
        return sorted(set(versions))

    def get_global_graph(self, domain: str, task: str, version: int | None = None) -> List[GlobalGraphItem]:
        versions = self.list_global_graph_versions(domain, task)
        if not versions:
            return []
        ver = version if version is not None else versions[-1]
        if ver not in versions:
            return []

        path = self._global_domain_dir(domain) / f"{self._task_slug(task)}_v{ver}.json"
        doc = read_json(path, default=None)
        if not isinstance(doc, dict):
            return []
        items = doc.get("items", [])
        if not isinstance(items, list):
            return []
        out: List[GlobalGraphItem] = []
        for it in items:
            if isinstance(it, dict):
                out.append(GlobalGraphItem.from_dict(it))
        return out

    def save_global_graph(self, domain: str, task: str, items: List[GlobalGraphItem]) -> int:
        versions = self.list_global_graph_versions(domain, task)
        new_ver = (versions[-1] + 1) if versions else 1

        d = self._global_domain_dir(domain)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self._task_slug(task)}_v{new_ver}.json"
        payload = {
            "domain": domain,
            "task": task,
            "version": new_ver,
            "created_at": datetime_to_iso(now_utc()),
            "items": [it.to_dict() for it in items],
        }
        write_json_atomic(path, payload, indent=2)
        return new_ver

