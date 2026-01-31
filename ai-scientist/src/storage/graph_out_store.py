from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from storage.json_storage import read_json, write_json_atomic


class GraphOutStore:
    """
    Read/write store for a `graph_out.json`-shaped local graph:

    {
      "nodes": [ { "id": "...", ... }, ... ],
      "edges": [ { "source": "...", "target": "...", "relation_text": "...", ... }, ... ]
    }
    """

    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Dict[str, Any]:
        doc = read_json(self.path, default=None)
        if not isinstance(doc, dict):
            doc = {}
        nodes = doc.get("nodes")
        edges = doc.get("edges")
        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []
        doc["nodes"] = nodes
        doc["edges"] = edges
        return doc

    def save(self, doc: Dict[str, Any]) -> None:
        write_json_atomic(self.path, doc, indent=2)

    def stats(self) -> Dict[str, int]:
        doc = self.load()
        return {"nodes": len(doc.get("nodes", [])), "edges": len(doc.get("edges", []))}

    @staticmethod
    def _node_id(node: Dict[str, Any]) -> Optional[str]:
        v = node.get("id") or node.get("paper_id")
        return str(v) if v else None

    @staticmethod
    def _edge_key(edge: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        src = edge.get("source") or edge.get("src_paper_id")
        dst = edge.get("target") or edge.get("dst_paper_id")
        return (str(src) if src else None, str(dst) if dst else None)

    def get_node(self, paper_id: str) -> Optional[Dict[str, Any]]:
        doc = self.load()
        for n in doc["nodes"]:
            if isinstance(n, dict) and self._node_id(n) == paper_id:
                return n
        return None

    def list_nodes(self) -> List[Dict[str, Any]]:
        doc = self.load()
        return [n for n in doc["nodes"] if isinstance(n, dict)]

    def upsert_node(self, node: Dict[str, Any]) -> None:
        nid = self._node_id(node)
        if not nid:
            raise ValueError("node must contain 'id' or 'paper_id'")
        doc = self.load()
        nodes = doc["nodes"]
        for i, n in enumerate(nodes):
            if isinstance(n, dict) and self._node_id(n) == nid:
                nodes[i] = node
                self.save(doc)
                return
        nodes.append(node)
        self.save(doc)

    def delete_node(self, paper_id: str) -> bool:
        doc = self.load()
        nodes = doc["nodes"]
        edges = doc["edges"]

        before_nodes = len(nodes)
        doc["nodes"] = [n for n in nodes if not (isinstance(n, dict) and self._node_id(n) == paper_id)]
        after_nodes = len(doc["nodes"])

        # Remove edges pointing to the deleted node as well
        def _keep_edge(e: Any) -> bool:
            if not isinstance(e, dict):
                return False
            src, dst = self._edge_key(e)
            return src != paper_id and dst != paper_id

        doc["edges"] = [e for e in edges if _keep_edge(e)]
        self.save(doc)
        return after_nodes < before_nodes

    def list_edges(self) -> List[Dict[str, Any]]:
        doc = self.load()
        return [e for e in doc["edges"] if isinstance(e, dict)]

    def upsert_edge(self, edge: Dict[str, Any]) -> None:
        src, dst = self._edge_key(edge)
        if not src or not dst:
            raise ValueError("edge must contain 'source'/'target' (or 'src_paper_id'/'dst_paper_id')")
        doc = self.load()
        edges = doc["edges"]
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                continue
            es, ed = self._edge_key(e)
            if es == src and ed == dst:
                edges[i] = edge
                self.save(doc)
                return
        edges.append(edge)
        self.save(doc)


