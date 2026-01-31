"""
Domain/Task-aware storage for Local Graph data.

Directory structure:
  data/{domain}/{task}/local/content/{paper_id}.json
  data/{domain}/{task}/local/implementation/{paper_id}.json
  data/{domain}/{task}/local/edge.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from storage.json_storage import slugify_component, write_json_atomic, read_json


# =============================================================================
# Domain/Task Mappings
# =============================================================================

# Map raw domain values to canonical domain names
DOMAIN_MAPPING = {
    "Multi_Modal_Recommendation": "Recsys",
    "General Recommendation": "Recsys",
    "Graph-based Collaborative Filtering": "Recsys",
    "Graph-based Recommendation": "Recsys",
    "Graph Neural Networks / Collaborative Filtering": "Recsys",
    "Multi-view Graph Representation Learning for Recommendation": "Recsys",
    "Recommendation Systems / Denoising Implicit Feedback": "Recsys",
    "Implicit Feedback Denoising": "Recsys",
    "Graph Neural Networks": "GraphLearning",
    "Deep Generative Models / Variational Inference": "MultiModal",
    "Multimodal Rumor Detection": "MultiModal",
}

# Map raw task values to canonical task names
TASK_MAPPING = {
    "Multi_Modal_Recommendation": "MultiModalRecommendation",
    "General Recommendation": "GeneralRecommendation",
    "Graph-based Collaborative Filtering": "GeneralRecommendation",
    "Graph-based Recommendation": "GeneralRecommendation",
    "Graph Neural Networks / Collaborative Filtering": "GeneralRecommendation",
    "Multi-view Graph Representation Learning for Recommendation": "MultiModalRecommendation",
    "Recommendation Systems / Denoising Implicit Feedback": "GeneralRecommendation",
    "Implicit Feedback Denoising": "GeneralRecommendation",
    "Multimodal Rumor Detection": "MultiModalRecommendation",
}


def normalize_domain(raw_domain: str) -> str:
    """Map raw domain to canonical domain name."""
    return DOMAIN_MAPPING.get(raw_domain, "Recsys")


def normalize_task(raw_task: str) -> str:
    """Map raw task to canonical task name."""
    return TASK_MAPPING.get(raw_task, "MultiModalRecommendation")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PaperContent:
    """Paper content node in local graph.
    
    Stored at: data/{domain}/{task}/local/content/{id}.json
    """
    id: str
    paper_title: str
    alias: str  # method_name
    year: int
    domain: str  # Recsys/CV/MultiModal/TimeSeries/GraphLearning
    task: str  # MultiModalRecommendation/GeneralRecommendation/...
    introduction: str = ""  # empty for now
    method: str = ""  # from method_md
    experiments: str = ""  # empty for now
    hyperparameter: str = ""  # from hyperparam_def
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "paper_title": self.paper_title,
            "alias": self.alias,
            "year": self.year,
            "domain": self.domain,
            "task": self.task,
            "introduction": self.introduction,
            "method": self.method,
            "experiments": self.experiments,
            "hyperparameter": self.hyperparameter,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperContent":
        return cls(
            id=str(d.get("id", "")),
            paper_title=str(d.get("paper_title", "")),
            alias=str(d.get("alias", d.get("method_name", ""))),
            year=int(d.get("year", 0) or 0),
            domain=str(d.get("domain", "")),
            task=str(d.get("task", "")),
            introduction=str(d.get("introduction", "")),
            method=str(d.get("method", d.get("method_md", ""))),
            experiments=str(d.get("experiments", "")),
            hyperparameter=str(d.get("hyperparameter", d.get("hyperparam_def", ""))),
        )
    
    @classmethod
    def from_graph_out_node(cls, node: Dict[str, Any]) -> "PaperContent":
        """Create from graph_out.json node."""
        raw_domain = str(node.get("domain", ""))
        raw_task = node.get("task", "")
        if isinstance(raw_task, list):
            raw_task = raw_task[0] if raw_task else ""
        
        return cls(
            id=str(node.get("id", "")),
            paper_title=str(node.get("paper_title", "")),
            alias=str(node.get("method_name", "")),
            year=int(node.get("year", 0) or 0),
            domain=normalize_domain(raw_domain),
            task=normalize_task(str(raw_task)),
            introduction="",  # empty for now
            method=str(node.get("method_md", "")),
            experiments="",  # empty for now
            hyperparameter=str(node.get("hyperparam_def", "")),
        )


@dataclass
class PaperImplementation:
    """Paper implementation in local graph.
    
    Stored at: data/{domain}/{task}/local/implementation/{id}/
        - meta.json          # metadata
        - algorithm.py       # Python code
        - hyperparameter.yaml # YAML config
    """
    id: str
    paper_title: str
    alias: str  # method_name
    year: int
    domain: str
    task: str
    algorithm: str = ""  # from model_code (full py content)
    hyperparameter: str = ""  # from config_yaml (full yaml content)
    
    def to_meta_dict(self) -> Dict[str, Any]:
        """Return metadata only (for meta.json)."""
        return {
            "id": self.id,
            "paper_title": self.paper_title,
            "alias": self.alias,
            "year": self.year,
            "domain": self.domain,
            "task": self.task,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full dict including code (for compatibility)."""
        d = self.to_meta_dict()
        d["algorithm"] = self.algorithm
        d["hyperparameter"] = self.hyperparameter
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperImplementation":
        return cls(
            id=str(d.get("id", "")),
            paper_title=str(d.get("paper_title", "")),
            alias=str(d.get("alias", "")),
            year=int(d.get("year", 0) or 0),
            domain=str(d.get("domain", "")),
            task=str(d.get("task", "")),
            algorithm=str(d.get("algorithm", "")),
            hyperparameter=str(d.get("hyperparameter", "")),
        )
    
    @classmethod
    def from_graph_out_node(cls, node: Dict[str, Any]) -> "PaperImplementation":
        """Create from graph_out.json node."""
        raw_domain = str(node.get("domain", ""))
        raw_task = node.get("task", "")
        if isinstance(raw_task, list):
            raw_task = raw_task[0] if raw_task else ""
        
        return cls(
            id=str(node.get("id", "")),
            paper_title=str(node.get("paper_title", "")),
            alias=str(node.get("method_name", "")),
            year=int(node.get("year", 0) or 0),
            domain=normalize_domain(raw_domain),
            task=normalize_task(str(raw_task)),
            algorithm=str(node.get("model_code", "")),
            hyperparameter=str(node.get("config_yaml", "")),
        )


@dataclass
class GraphEdge:
    """Edge in local graph.
    
    Stored in: data/{domain}/{task}/local/edge.json (list)
    """
    source: str  # paper_id
    target: str  # paper_id
    type: str  # "in-domain" or "out-of-domain"
    # For in-domain edges
    similarities: Optional[str] = None
    differences: Optional[str] = None
    # For out-of-domain edges
    relation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
        }
        if self.type == "in-domain":
            d["similarities"] = self.similarities
            d["differences"] = self.differences
        else:
            d["relation"] = self.relation
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdge":
        return cls(
            source=str(d.get("source", "")),
            target=str(d.get("target", "")),
            type=str(d.get("type", "")),
            similarities=d.get("similarities"),
            differences=d.get("differences"),
            relation=d.get("relation"),
        )


# =============================================================================
# Storage Classes
# =============================================================================

class DomainTaskStorage:
    """
    Storage manager for domain/task-aware local graph data.
    
    Directory structure:
      data_root/{domain}/{task}/local/content/{paper_id}.json
      data_root/{domain}/{task}/local/implementation/{paper_id}.json
      data_root/{domain}/{task}/local/edge.json
    """
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
    
    def _local_dir(self, domain: str, task: str) -> Path:
        """Get the local directory for a domain/task."""
        return self.data_root / slugify_component(domain) / slugify_component(task) / "local"
    
    def _content_dir(self, domain: str, task: str) -> Path:
        return self._local_dir(domain, task) / "content"
    
    def _implementation_dir(self, domain: str, task: str) -> Path:
        return self._local_dir(domain, task) / "implementation"
    
    def _edge_file(self, domain: str, task: str) -> Path:
        return self._local_dir(domain, task) / "edge.json"
    
    # =========== Content Operations ===========
    
    def save_content(self, content: PaperContent) -> Path:
        """Save paper content to file."""
        content_dir = self._content_dir(content.domain, content.task)
        content_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = content_dir / f"{slugify_component(content.id)}.json"
        write_json_atomic(file_path, content.to_dict(), indent=2)
        return file_path
    
    def get_content(self, domain: str, task: str, paper_id: str) -> Optional[PaperContent]:
        """Get paper content by id."""
        content_dir = self._content_dir(domain, task)
        file_path = content_dir / f"{slugify_component(paper_id)}.json"
        
        data = read_json(file_path, default=None)
        if isinstance(data, dict):
            return PaperContent.from_dict(data)
        return None
    
    def list_contents(self, domain: str, task: str) -> List[PaperContent]:
        """List all paper contents for a domain/task."""
        content_dir = self._content_dir(domain, task)
        if not content_dir.exists():
            return []
        
        result = []
        for file_path in sorted(content_dir.glob("*.json")):
            data = read_json(file_path, default=None)
            if isinstance(data, dict):
                result.append(PaperContent.from_dict(data))
        return result
    
    # =========== Implementation Operations ===========
    
    def save_implementation(self, impl: PaperImplementation) -> Path:
        """Save paper implementation to separate files for better readability.
        
        Creates:
            {impl_dir}/{paper_id}/meta.json          # metadata
            {impl_dir}/{paper_id}/algorithm.py       # Python code
            {impl_dir}/{paper_id}/hyperparameter.yaml # YAML config
        """
        impl_dir = self._implementation_dir(impl.domain, impl.task)
        paper_dir = impl_dir / slugify_component(impl.id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # Save meta.json
        meta_path = paper_dir / "meta.json"
        write_json_atomic(meta_path, impl.to_meta_dict(), indent=2)
        
        # Save algorithm.py (actual Python file)
        algo_path = paper_dir / "algorithm.py"
        algo_path.write_text(impl.algorithm, encoding="utf-8")
        
        # Save hyperparameter.yaml (actual YAML file)
        hp_path = paper_dir / "hyperparameter.yaml"
        hp_path.write_text(impl.hyperparameter, encoding="utf-8")
        
        return paper_dir
    
    def get_implementation(self, domain: str, task: str, paper_id: str) -> Optional[PaperImplementation]:
        """Get paper implementation by id."""
        impl_dir = self._implementation_dir(domain, task)
        paper_dir = impl_dir / slugify_component(paper_id)
        meta_path = paper_dir / "meta.json"
        
        data = read_json(meta_path, default=None)
        if not isinstance(data, dict):
            return None
        
        # Read algorithm.py
        algo_path = paper_dir / "algorithm.py"
        algorithm = algo_path.read_text(encoding="utf-8") if algo_path.exists() else ""
        
        # Read hyperparameter.yaml
        hp_path = paper_dir / "hyperparameter.yaml"
        hyperparameter = hp_path.read_text(encoding="utf-8") if hp_path.exists() else ""
        
        return PaperImplementation(
            id=str(data.get("id", "")),
            paper_title=str(data.get("paper_title", "")),
            alias=str(data.get("alias", "")),
            year=int(data.get("year", 0) or 0),
            domain=str(data.get("domain", "")),
            task=str(data.get("task", "")),
            algorithm=algorithm,
            hyperparameter=hyperparameter,
        )
    
    def list_implementations(self, domain: str, task: str) -> List[PaperImplementation]:
        """List all implementations for a domain/task."""
        impl_dir = self._implementation_dir(domain, task)
        if not impl_dir.exists():
            return []
        
        result = []
        for paper_dir in sorted(impl_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            meta_path = paper_dir / "meta.json"
            if not meta_path.exists():
                continue
            
            impl = self.get_implementation(domain, task, paper_dir.name)
            if impl:
                result.append(impl)
        return result
    
    # =========== Edge Operations ===========
    
    def save_edges(self, domain: str, task: str, edges: List[GraphEdge]) -> Path:
        """Save all edges for a domain/task."""
        local_dir = self._local_dir(domain, task)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = self._edge_file(domain, task)
        data = [e.to_dict() for e in edges]
        write_json_atomic(file_path, data, indent=2)
        return file_path
    
    def get_edges(self, domain: str, task: str) -> List[GraphEdge]:
        """Get all edges for a domain/task."""
        file_path = self._edge_file(domain, task)
        data = read_json(file_path, default=None)
        
        if isinstance(data, list):
            return [GraphEdge.from_dict(e) for e in data if isinstance(e, dict)]
        return []
    
    def add_edge(self, domain: str, task: str, edge: GraphEdge) -> None:
        """Add a single edge (append to existing)."""
        edges = self.get_edges(domain, task)
        # Check for duplicates
        for e in edges:
            if e.source == edge.source and e.target == edge.target:
                return  # Already exists
        edges.append(edge)
        self.save_edges(domain, task, edges)


def parse_relation_text(text: str) -> Dict[str, Optional[str]]:
    """
    Parse relation_text from graph_out.json to extract similarities and differences.
    
    Returns dict with 'similarities', 'differences', 'relation' keys.
    """
    if not text:
        return {"similarities": None, "differences": None, "relation": None}
    
    # Check if text contains explicit Similarities/Differences sections
    text_lower = text.lower()
    has_similarities = "similarities:" in text_lower or "similarities\n" in text_lower
    has_differences = "differences:" in text_lower or "differences\n" in text_lower
    
    if has_similarities or has_differences:
        similarities = None
        differences = None
        
        # Try to extract similarities section
        import re
        sim_match = re.search(r'similarities:?\s*(.*?)(?=differences:|$)', text, re.IGNORECASE | re.DOTALL)
        diff_match = re.search(r'differences:?\s*(.*?)$', text, re.IGNORECASE | re.DOTALL)
        
        if sim_match:
            similarities = sim_match.group(1).strip()
        if diff_match:
            differences = diff_match.group(1).strip()
        
        return {
            "similarities": similarities or None,
            "differences": differences or None,
            "relation": None,
        }
    else:
        # No explicit sections, treat whole text as relation
        return {
            "similarities": None,
            "differences": None,
            "relation": text,
        }


def convert_graph_out_edge(
    edge: Dict[str, Any],
    node_tasks: Dict[str, str],  # paper_id -> task
) -> GraphEdge:
    """
    Convert a graph_out.json edge to GraphEdge.
    
    Note: The edge direction in graph_out.json is REVERSED.
    Original: source -> target means "source cites target"
    We want: source -> target means "source is prior to target" (follow-up direction)
    
    So we swap source and target.
    """
    # Swap direction: graph_out has reversed edges
    original_source = str(edge.get("source", ""))
    original_target = str(edge.get("target", ""))
    
    # After swap: new_source is the older paper, new_target is the follow-up
    new_source = original_target
    new_target = original_source
    
    # Determine edge type based on task
    source_task = node_tasks.get(new_source, "")
    target_task = node_tasks.get(new_target, "")
    
    edge_type = "in-domain" if source_task == target_task else "out-of-domain"
    
    # Parse relation_text
    relation_text = str(edge.get("relation_text", ""))
    parsed = parse_relation_text(relation_text)
    
    return GraphEdge(
        source=new_source,
        target=new_target,
        type=edge_type,
        similarities=parsed["similarities"],
        differences=parsed["differences"],
        relation=parsed["relation"],
    )

