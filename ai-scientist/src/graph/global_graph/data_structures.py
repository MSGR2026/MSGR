"""
Global Graph 数据结构

定义全局图的核心数据结构:
    - GlobalGraphItem: 单个知识项
    - GlobalGraph: 知识项集合

作者: GraphScientist Team
版本: 1.0.0
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class GlobalGraphItem:
    """
    全局图知识项
    
    表示一条领域知识，包含标题、内容和元数据。
    
    属性:
        id: 唯一标识符 (自动生成)
        title: 知识项标题
        content: 知识项内容
    
    使用示例:
        >>> item = GlobalGraphItem(
        ...     title="多模态特征融合策略",
        ...     content="常见的多模态特征融合策略包括..."
        ... )
    """
    
    title: str                          # 知识项标题
    content: str                        # 知识项内容
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])  # 唯一ID
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        返回:
            包含所有属性的字典
        """
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalGraphItem":
        """
        从字典创建实例
        
        参数:
            data: 包含属性的字典
        
        返回:
            GlobalGraphItem 实例
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            title=data.get("title", ""),
            content=data.get("content", ""),
        )
    
    def update(self, **kwargs) -> None:
        """
        更新属性
        
        参数:
            **kwargs: 要更新的属性
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_context_string(self) -> str:
        """
        转换为上下文字符串，用于 Prompt 构建
        
        返回:
            格式化的字符串
        """
        lines = [f"### {self.title}"]
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)


@dataclass
class GlobalGraph:
    """
    全局领域知识图谱
    
    管理一组 GlobalGraphItem，提供增删改查操作。
    
    属性:
        domain: 领域名称 (如 "Recsys")
        task: 任务名称 (如 "MultiModal")
        items: 知识项列表
        metadata: 元数据字典
    
    使用示例:
        >>> graph = GlobalGraph(domain="Recsys", task="MultiModal")
        >>> graph.add_item(GlobalGraphItem(title="...", content="..."))
        >>> context = graph.to_context_string()
    """
    
    domain: str                         # 领域名称
    task: str                           # 任务名称
    items: List[GlobalGraphItem] = field(default_factory=list)  # 知识项列表
    metadata: Dict[str, Any] = field(default_factory=dict)      # 元数据
    
    def __post_init__(self):
        """初始化后处理"""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
        self.metadata["updated_at"] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        返回:
            包含所有属性的字典
        """
        return {
            "domain": self.domain,
            "task": self.task,
            "items": [item.to_dict() for item in self.items],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalGraph":
        """
        从字典创建实例
        
        参数:
            data: 包含属性的字典
        
        返回:
            GlobalGraph 实例
        """
        items = [GlobalGraphItem.from_dict(item) for item in data.get("items", [])]
        return cls(
            domain=data.get("domain", ""),
            task=data.get("task", ""),
            items=items,
            metadata=data.get("metadata", {}),
        )
    
    # ==================== 增删改查操作 ====================
    
    def add_item(self, item: GlobalGraphItem) -> None:
        """
        添加知识项
        
        参数:
            item: 要添加的知识项
        """
        # 检查是否已存在相同 ID
        if any(existing.id == item.id for existing in self.items):
            raise ValueError(f"知识项 ID 已存在: {item.id}")
        self.items.append(item)
        self._update_metadata()
    
    def remove_item(self, item_id: str) -> bool:
        """
        移除知识项
        
        参数:
            item_id: 要移除的知识项 ID
        
        返回:
            是否成功移除
        """
        for i, item in enumerate(self.items):
            if item.id == item_id:
                self.items.pop(i)
                self._update_metadata()
                return True
        return False
    
    def get_item(self, item_id: str) -> Optional[GlobalGraphItem]:
        """
        根据 ID 获取知识项
        
        参数:
            item_id: 知识项 ID
        
        返回:
            知识项，如果不存在则返回 None
        """
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def get_item_by_title(self, title: str) -> Optional[GlobalGraphItem]:
        """
        根据标题获取知识项
        
        参数:
            title: 知识项标题
        
        返回:
            知识项，如果不存在则返回 None
        """
        for item in self.items:
            if item.title == title:
                return item
        return None
    
    def update_item(self, item_id: str, **kwargs) -> bool:
        """
        更新知识项
        
        参数:
            item_id: 知识项 ID
            **kwargs: 要更新的属性
        
        返回:
            是否成功更新
        """
        item = self.get_item(item_id)
        if item:
            item.update(**kwargs)
            self._update_metadata()
            return True
        return False
    
    def list_items(self) -> List[GlobalGraphItem]:
        """
        列出知识项

        返回:
            符合条件的知识项列表
        """
        return list(self.items)
    
    # ==================== 上下文生成 ====================
    
    def to_context_string(
        self,
        max_items: int = 10,
    ) -> str:
        """
        转换为上下文字符串，用于 Prompt 构建
        
        参数:
            max_items: 最大知识项数量
        
        返回:
            格式化的上下文字符串
        """
        items = self.list_items()[:max_items]
        
        if not items:
            return ""
        
        lines = [f"## {self.domain} - {self.task} 领域知识\n"]
        for item in items:
            lines.append(item.to_context_string())
            lines.append("")
        
        return "\n".join(lines)
    
    # ==================== 统计信息 ====================
    
    def __len__(self) -> int:
        """返回知识项数量"""
        return len(self.items)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        返回:
            统计信息字典
        """
        return {
            "total_items": len(self.items),
            "domain": self.domain,
            "task": self.task,
        }
    
    # ==================== 私有方法 ====================
    
    def _update_metadata(self) -> None:
        """更新元数据"""
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.metadata["item_count"] = len(self.items)
