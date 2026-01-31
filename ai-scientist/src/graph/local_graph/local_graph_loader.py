"""
LocalGraphLoader - Interface for loading local graph data.

Provides methods to access paper content, implementations, and neighbor information
from the domain/task-aware storage structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config import get_repo_root, load_yaml, resolve_data_root
from storage.domain_task_storage import (
    DomainTaskStorage,
    GraphEdge,
    PaperContent,
    PaperImplementation,
)
from storage.json_storage import slugify_component


@dataclass
class NeighborInfo:
    """Information about a neighbor paper and its edge."""
    paper_id: str
    paper_title: str
    alias: str
    year: int
    method: str
    introduction: str
    algorithm_implementation: str
    hyperparameter_implementation: str
    # Edge semantic information
    edge_type: str  # "in-domain" or "out-of-domain"
    similarities: Optional[str] = None  # For in-domain edges
    differences: Optional[str] = None   # For in-domain edges
    relation: Optional[str] = None      # For out-of-domain edges
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "alias": self.alias,
            "year": self.year,
            "method": self.method,
            "introduction": self.introduction,
            "algorithm_implementation": self.algorithm_implementation,
            "hyperparameter_implementation": self.hyperparameter_implementation,
            "edge_type": self.edge_type,
            "similarities": self.similarities,
            "differences": self.differences,
            "relation": self.relation,
        }


class LocalGraphLoader:
    """
    Loader for local graph data.
    
    Provides access to paper content, implementations, and neighbor information
    for a specific domain/task combination.
    """
    
    def __init__(
        self,
        domain: str = "Recsys",
        task: str = "MultiModal",
        data_root: Optional[Path] = None,
    ):
        """
        Initialize the LocalGraphLoader.
        
        Args:
            domain: The domain to load (e.g., "Recsys", "CV", "MultiModal")
            task: The task to load (e.g., "MultiModalRecommendation", "GeneralRecommendation")
            data_root: Optional custom data root path. If None, uses default from config.
        """
        self.domain = domain
        self.task = task
        
        if data_root is None:
            data_root = self._get_default_data_root()
        
        self.data_root = Path(data_root)
        self.storage = DomainTaskStorage(self.data_root)
        
        # Cache for content and edges
        self._content_cache: Dict[str, PaperContent] = {}
        self._impl_cache: Dict[str, PaperImplementation] = {}
        self._edges_cache: Optional[List[GraphEdge]] = None
        self._all_edges_cache: Optional[List[GraphEdge]] = None

    def _get_neighbor_override_impl(
        self,
        target_paper_id: str,
        neighbor_paper_id: str,
        domain: str,
        task: str,
    ) -> Optional[PaperImplementation]:
        """Load neighbor implementation override from target paper folder."""
        impl_dir = self.storage._implementation_dir(domain, task)
        neighbor_dir = (
            impl_dir
            / slugify_component(target_paper_id)
            / "neighbors"
            / slugify_component(neighbor_paper_id)
        )
        algo_path = neighbor_dir / "algorithm.py"
        hp_path = neighbor_dir / "hyperparameter.yaml"

        if not algo_path.exists() and not hp_path.exists():
            return None

        algorithm = algo_path.read_text(encoding="utf-8") if algo_path.exists() else ""
        hyperparameter = hp_path.read_text(encoding="utf-8") if hp_path.exists() else ""

        content = self._get_content(neighbor_paper_id)
        if content is None:
            content = self._find_content_anywhere(neighbor_paper_id)

        return PaperImplementation(
            id=neighbor_paper_id,
            paper_title=content.paper_title if content else "",
            alias=content.alias if content else "",
            year=content.year if content else 0,
            domain=domain,
            task=task,
            algorithm=algorithm,
            hyperparameter=hyperparameter,
        )
    
    def _get_default_data_root(self) -> Path:
        """Get default data root from config."""
        repo_root = get_repo_root()
        config_path = repo_root / "ai-scientist" / "configs" / "storage.yaml"
        if config_path.exists():
            cfg = load_yaml(config_path)
            return resolve_data_root(cfg.get("data_root"))
        return repo_root / "data"
    
    def _get_content(self, paper_id: str) -> Optional[PaperContent]:
        """Get paper content with caching."""
        if paper_id not in self._content_cache:
            content = self.storage.get_content(self.domain, self.task, paper_id)
            if content:
                self._content_cache[paper_id] = content
        return self._content_cache.get(paper_id)
    
    def _get_implementation(self, paper_id: str) -> Optional[PaperImplementation]:
        """Get paper implementation with caching."""
        if paper_id not in self._impl_cache:
            impl = self.storage.get_implementation(self.domain, self.task, paper_id)
            if impl:
                self._impl_cache[paper_id] = impl
        return self._impl_cache.get(paper_id)
    
    def _get_edges(self) -> List[GraphEdge]:
        """Get all edges for this domain/task with caching."""
        if self._edges_cache is None:
            self._edges_cache = self.storage.get_edges(self.domain, self.task)
        return self._edges_cache
    
    def _get_all_edges(self) -> List[GraphEdge]:
        """Get edges from all domain/task combinations (for cross-domain lookups)."""
        if self._all_edges_cache is None:
            all_edges = []
            # Scan all domains and tasks
            for domain_dir in self.data_root.iterdir():
                if not domain_dir.is_dir():
                    continue
                for task_dir in domain_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                    edges = self.storage.get_edges(domain_dir.name, task_dir.name)
                    all_edges.extend(edges)
            self._all_edges_cache = all_edges
        return self._all_edges_cache
    
    # =========== Content Access Methods ===========
    
    def get_method(self, paper_id: str) -> str:
        """
        Get the method section content of a paper.
        
        Args:
            paper_id: The paper ID (e.g., "BM3_2023")
            
        Returns:
            The method content string, or empty string if not found.
        """
        content = self._get_content(paper_id)
        return content.method if content else ""
    
    def get_introduction(self, paper_id: str) -> str:
        """
        Get the introduction section of a paper.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            The introduction content string, or empty string if not found.
        """
        content = self._get_content(paper_id)
        return content.introduction if content else ""
    
    def get_hyperparameter(self, paper_id: str) -> str:
        """
        Get the hyperparameter description of a paper (from the paper's definition).
        
        This is the hyperparameter section from the paper, not the actual YAML config.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            The hyperparameter description string, or empty string if not found.
        """
        content = self._get_content(paper_id)
        return content.hyperparameter if content else ""
    
    def get_idea(self, paper_id: str) -> str:
        """
        Get the idea/contribution of a paper.
        
        Note: Currently stored in method field. Will be separated in future.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            The idea content string, or empty string if not found.
        """
        # Note: idea is not explicitly stored in current schema
        # We return method as a fallback since idea is often part of method description
        content = self._get_content(paper_id)
        return content.method if content else ""
    
    # =========== Implementation Access Methods ===========
    
    def get_hyperparameter_implementation(self, paper_id: str) -> str:
        """
        Get the actual hyperparameter YAML configuration of a paper's implementation.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            The hyperparameter YAML content, or empty string if not found.
        """
        impl = self._get_implementation(paper_id)
        return impl.hyperparameter if impl else ""
    
    def get_algorithm_implementation(self, paper_id: str) -> str:
        """
        Get the actual algorithm code of a paper's implementation.
        
        Args:
            paper_id: The paper ID
            
        Returns:
            The algorithm Python code, or empty string if not found.
        """
        impl = self._get_implementation(paper_id)
        return impl.algorithm if impl else ""
    
    # =========== Graph Navigation Methods ===========
    
    def get_neighbors(
        self,
        paper_id: str,
        k: int = 1,
        edge_type: Optional[str] = None,
    ) -> List[NeighborInfo]:
        """
        Get the K-hop neighbors (prior baselines) of a paper.
        
        This returns papers that the given paper follows/cites, including their
        content, implementations, and edge semantic information.
        
        Args:
            paper_id: The paper ID to find neighbors for
            k: The number of hops (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
            edge_type: Filter by edge type ("in-domain" or "out-of-domain"). 
                      If None, returns both types.
        
        Returns:
            List of NeighborInfo containing neighbor papers' content and edge info.
        """
        if k < 1:
            return []

        base_domain = self.domain
        base_task = self.task
        base_content = self._get_content(paper_id)
        if base_content is None:
            base_content = self._find_content_anywhere(paper_id)
        if base_content:
            if base_content.domain:
                base_domain = base_content.domain
            if base_content.task:
                base_task = base_content.task

        # Use all edges for cross-domain lookups
        all_edges = self._get_all_edges()
        
        # Build adjacency: paper_id -> list of (neighbor_id, edge)
        # We look for edges where target == paper_id (i.e., paper_id follows source)
        adjacency: Dict[str, List[GraphEdge]] = {}
        for edge in all_edges:
            if edge.target not in adjacency:
                adjacency[edge.target] = []
            adjacency[edge.target].append(edge)
        
        # BFS to find K-hop neighbors
        visited: set = set()
        current_level = {paper_id}
        neighbor_edges: List[GraphEdge] = []
        
        for hop in range(k):
            next_level: set = set()
            for pid in current_level:
                if pid in visited:
                    continue
                visited.add(pid)
                
                for edge in adjacency.get(pid, []):
                    # Filter by edge type if specified
                    if edge_type is not None and edge.type != edge_type:
                        continue
                    
                    neighbor_id = edge.source
                    if neighbor_id not in visited:
                        next_level.add(neighbor_id)
                        neighbor_edges.append(edge)
            
            current_level = next_level
        
        # Build NeighborInfo for each neighbor
        result: List[NeighborInfo] = []
        seen_neighbors: set = set()
        
        for edge in neighbor_edges:
            neighbor_id = edge.source
            if neighbor_id in seen_neighbors:
                continue
            seen_neighbors.add(neighbor_id)
            
            # Try to get content and implementation from current domain/task first
            content = self._get_content(neighbor_id)
            impl = self._get_neighbor_override_impl(
                target_paper_id=paper_id,
                neighbor_paper_id=neighbor_id,
                domain=base_domain,
                task=base_task,
            )
            if impl is None:
                impl = self.storage.get_implementation(base_domain, base_task, neighbor_id)
                if impl is None:
                    impl = self._get_implementation(neighbor_id)
            
            # If not found in current domain/task, search all domains
            if content is None:
                content = self._find_content_anywhere(neighbor_id)
            if impl is None:
                impl = self._find_implementation_anywhere(neighbor_id)
            
            result.append(NeighborInfo(
                paper_id=neighbor_id,
                paper_title=content.paper_title if content else "",
                alias=content.alias if content else "",
                year=content.year if content else 0,
                method=content.method if content else "",
                introduction=content.introduction if content else "",
                algorithm_implementation=impl.algorithm if impl else "",
                hyperparameter_implementation=impl.hyperparameter if impl else "",
                edge_type=edge.type,
                similarities=edge.similarities,
                differences=edge.differences,
                relation=edge.relation,
            ))
        
        return result
    
    def _find_content_anywhere(self, paper_id: str) -> Optional[PaperContent]:
        """Search for content in all domains/tasks."""
        for domain_dir in self.data_root.iterdir():
            if not domain_dir.is_dir():
                continue
            for task_dir in domain_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                content = self.storage.get_content(domain_dir.name, task_dir.name, paper_id)
                if content:
                    return content
        return None
    
    def _find_implementation_anywhere(self, paper_id: str) -> Optional[PaperImplementation]:
        """Search for implementation in all domains/tasks."""
        for domain_dir in self.data_root.iterdir():
            if not domain_dir.is_dir():
                continue
            for task_dir in domain_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                impl = self.storage.get_implementation(domain_dir.name, task_dir.name, paper_id)
                if impl:
                    return impl
        return None
    
    # =========== Utility Methods ===========
    
    def list_papers(self) -> List[str]:
        """List all paper IDs in the current domain/task."""
        contents = self.storage.list_contents(self.domain, self.task)
        return [c.id for c in contents]
    
    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get full paper information including content and implementation."""
        content = self._get_content(paper_id)
        impl = self._get_implementation(paper_id)
        
        if content is None:
            return None
        
        return {
            "id": content.id,
            "paper_title": content.paper_title,
            "alias": content.alias,
            "year": content.year,
            "domain": content.domain,
            "task": content.task,
            "method": content.method,
            "introduction": content.introduction,
            "hyperparameter": content.hyperparameter,
            "has_implementation": impl is not None,
            "algorithm_implementation": impl.algorithm if impl else "",
            "hyperparameter_implementation": impl.hyperparameter if impl else "",
        }
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._content_cache.clear()
        self._impl_cache.clear()
        self._edges_cache = None
        self._all_edges_cache = None
