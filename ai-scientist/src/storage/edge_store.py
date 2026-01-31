from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.types import SemanticEdge, datetime_to_iso, now_utc
from storage.json_storage import read_json, write_json_atomic


class EdgeStore:
    """
    Local graph Edge storage (file-backed JSON).

    Layout (requested):
      data_root/graphs/local/edge/edges.json
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.edge_dir = self.data_root / "graphs" / "local" / "edge"
        self.edges_file = self.edge_dir / "edges.json"

    def _load_doc(self) -> Dict[str, Any]:
        doc = read_json(self.edges_file, default=None)
        if not isinstance(doc, dict):
            doc = {}
        if "edges" not in doc or not isinstance(doc.get("edges"), list):
            doc["edges"] = []
        return doc

    def _save_doc(self, doc: Dict[str, Any]) -> None:
        doc["updated_at"] = datetime_to_iso(now_utc())
        write_json_atomic(self.edges_file, doc, indent=2)

    def list_edges(self) -> List[SemanticEdge]:
        doc = self._load_doc()
        out: List[SemanticEdge] = []
        for e in doc.get("edges", []):
            if isinstance(e, dict):
                out.append(SemanticEdge.from_dict(e))
        return out

    def overwrite_edges(self, edges: List[SemanticEdge]) -> None:
        doc = {"edges": [e.to_dict() for e in edges]}
        self._save_doc(doc)

    def upsert_edge(self, edge: SemanticEdge) -> None:
        doc = self._load_doc()
        edges = doc["edges"]

        key = (edge.src_paper_id, edge.dst_paper_id, edge.edge_type)
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                continue
            k = (e.get("src_paper_id"), e.get("dst_paper_id"), e.get("edge_type"))
            if k == key:
                edges[i] = edge.to_dict()
                self._save_doc(doc)
                return
        edges.append(edge.to_dict())
        self._save_doc(doc)


