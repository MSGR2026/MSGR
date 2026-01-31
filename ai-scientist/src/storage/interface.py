from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from common.types import Content, GlobalGraphItem, Implementation, SemanticEdge


class GraphStorageInterface(ABC):
    """图存储抽象接口（Local Graph + Global Graph）。"""

    # ============ Content 操作 ============
    @abstractmethod
    def get_content(self, paper_id: str) -> Optional[Content]:
        """获取论文内容"""

    @abstractmethod
    def save_content(self, content: Content) -> None:
        """保存论文内容"""

    @abstractmethod
    def list_contents(self, domain: str | None = None, task: str | None = None) -> List[Content]:
        """列出论文内容"""

    @abstractmethod
    def delete_content(self, paper_id: str) -> bool:
        """删除论文内容"""

    # ============ Implementation 操作 ============
    @abstractmethod
    def get_implementation(self, impl_id: str) -> Optional[Implementation]:
        """获取实现"""

    @abstractmethod
    def get_ground_truth(self, paper_id: str) -> Optional[Implementation]:
        """获取官方实现"""

    @abstractmethod
    def save_implementation(self, impl: Implementation) -> None:
        """保存实现"""

    @abstractmethod
    def list_implementations(
        self, paper_id: str | None = None, session_id: str | None = None
    ) -> List[Implementation]:
        """列出实现"""

    # ============ Edge 操作 ============
    @abstractmethod
    def get_neighbors(self, paper_id: str, k: int = 5, edge_type: str | None = None) -> List[SemanticEdge]:
        """获取邻居论文（默认按 weight 降序截断 top-k）"""

    @abstractmethod
    def save_edge(self, edge: SemanticEdge) -> None:
        """保存语义边"""

    @abstractmethod
    def get_edges(self, domain: str) -> List[SemanticEdge]:
        """获取领域所有边"""

    # ============ Global Graph 操作 ============
    @abstractmethod
    def get_global_graph(self, domain: str, task: str, version: int | None = None) -> List[GlobalGraphItem]:
        """获取 Global Graph（默认最新版本）"""

    @abstractmethod
    def save_global_graph(self, domain: str, task: str, items: List[GlobalGraphItem]) -> int:
        """保存 Global Graph，返回新版本号"""

    @abstractmethod
    def list_global_graph_versions(self, domain: str, task: str) -> List[int]:
        """列出所有版本"""

