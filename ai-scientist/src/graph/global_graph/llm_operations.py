"""
Global Graph LLM 操作模块

提供由 LLM Agent 驱动的全局图操作功能。

操作类型:
    - ADD: 添加新知识项 (需要 title 和 content)
    - MERGE: 合并多个知识项 (需要指定源项和合并结果)
    - DELETE: 删除知识项 (需要指定项 ID)
    - MODIFY: 修改知识项 (需要指定项 ID 和修改内容)

工作流程:
    1. 构建操作提示词 (通过 PromptBuilder)
    2. 调用 LLM 生成操作指令
    3. 解析 LLM 响应
    4. 执行对应的图操作
    5. 验证操作结果

作者: GraphScientist Team
版本: 1.0.0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from graph.global_graph.data_structures import GlobalGraph, GlobalGraphItem

if TYPE_CHECKING:
    pass  # 避免循环导入


class OperationType(Enum):
    """
    操作类型枚举
    
    定义 LLM 可以执行的图操作类型。
    """
    ADD = "add"          # 添加新知识项
    MERGE = "merge"      # 合并多个知识项
    DELETE = "delete"    # 删除知识项
    MODIFY = "modify"    # 修改知识项


@dataclass
class OperationResult:
    """
    操作结果
    
    封装单次操作的执行结果。
    
    属性:
        success: 操作是否成功
        operation_type: 操作类型
        message: 结果消息
        item_id: 相关的知识项 ID (可选)
        data: 额外数据 (可选)
    """
    success: bool
    operation_type: OperationType
    message: str
    item_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "operation_type": self.operation_type.value,
            "message": self.message,
            "item_id": self.item_id,
            "data": self.data,
        }


@dataclass
class AddOperation:
    """
    添加操作
    
    添加新的知识项到全局图。
    
    属性:
        title: 知识项标题 (必需)
        content: 知识项内容 (必需)
    """
    title: str
    content: str
    
    def validate(self) -> tuple[bool, str]:
        """
        验证操作参数
        
        返回:
            (是否有效, 错误消息)
        """
        if not self.title or not self.title.strip():
            return False, "标题不能为空"
        if not self.content or not self.content.strip():
            return False, "内容不能为空"
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operation": "add",
            "title": self.title,
            "content": self.content,
        }


@dataclass
class MergeOperation:
    """
    合并操作
    
    将多个知识项合并为一个新项。
    
    属性:
        source_ids: 要合并的知识项 ID 列表 (至少 2 个)
        merged_title: 合并后的标题 (必需)
        merged_content: 合并后的内容 (必需)
        delete_sources: 是否删除源项 (默认 True)
    """
    source_ids: List[str]
    merged_title: str
    merged_content: str
    delete_sources: bool = True
    
    def validate(self) -> tuple[bool, str]:
        """
        验证操作参数
        
        返回:
            (是否有效, 错误消息)
        """
        if len(self.source_ids) < 2:
            return False, "合并操作至少需要 2 个源知识项"
        if not self.merged_title or not self.merged_title.strip():
            return False, "合并后的标题不能为空"
        if not self.merged_content or not self.merged_content.strip():
            return False, "合并后的内容不能为空"
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operation": "merge",
            "source_ids": self.source_ids,
            "merged_title": self.merged_title,
            "merged_content": self.merged_content,
            "delete_sources": self.delete_sources,
        }


@dataclass
class DeleteOperation:
    """
    删除操作
    
    从全局图中删除知识项。
    
    属性:
        item_id: 要删除的知识项 ID (必需)
        reason: 删除原因 (可选，用于日志记录)
    """
    item_id: str
    reason: Optional[str] = None
    
    def validate(self) -> tuple[bool, str]:
        """
        验证操作参数
        
        返回:
            (是否有效, 错误消息)
        """
        if not self.item_id or not self.item_id.strip():
            return False, "知识项 ID 不能为空"
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operation": "delete",
            "item_id": self.item_id,
            "reason": self.reason,
        }


@dataclass
class ModifyOperation:
    """
    修改操作
    
    修改已有知识项的内容。
    
    属性:
        item_id: 要修改的知识项 ID (必需)
        new_title: 新标题 (可选，None 表示不修改)
        new_content: 新内容 (可选，None 表示不修改)
    """
    item_id: str
    new_title: Optional[str] = None
    new_content: Optional[str] = None
    
    def validate(self) -> tuple[bool, str]:
        """
        验证操作参数
        
        返回:
            (是否有效, 错误消息)
        """
        if not self.item_id or not self.item_id.strip():
            return False, "知识项 ID 不能为空"
        if all(v is None for v in [self.new_title, self.new_content]):
            return False, "至少需要指定一个要修改的字段"
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "operation": "modify",
            "item_id": self.item_id,
        }
        if self.new_title is not None:
            result["new_title"] = self.new_title
        if self.new_content is not None:
            result["new_content"] = self.new_content
        return result


class GlobalGraphLLMOperator:
    """
    Global Graph LLM 操作器
    
    提供 LLM 驱动的全局图操作功能。
    
    主要功能:
        1. 解析 LLM 响应为操作对象
        2. 执行图操作
        3. 验证操作结果
    
    使用示例:
        >>> operator = GlobalGraphLLMOperator(graph)
        >>> 
        >>> # 解析 LLM 响应
        >>> operations = operator.parse_llm_response(llm_output)
        >>> 
        >>> # 执行操作
        >>> results = operator.execute_operations(operations)
    """
    
    def __init__(self, graph: GlobalGraph):
        """
        初始化操作器
        
        参数:
            graph: 要操作的全局图实例
        """
        self.graph = graph
    
    # ==================== 操作执行 ====================
    
    def execute_add(self, op: AddOperation) -> OperationResult:
        """
        执行添加操作
        
        参数:
            op: 添加操作对象
        
        返回:
            操作结果
        """
        # 验证参数
        valid, error = op.validate()
        if not valid:
            return OperationResult(
                success=False,
                operation_type=OperationType.ADD,
                message=f"参数验证失败: {error}"
            )
        
        try:
            # 创建新知识项
            item = GlobalGraphItem(
                title=op.title.strip(),
                content=op.content.strip(),
            )
            
            # 添加到图
            self.graph.add_item(item)
            
            return OperationResult(
                success=True,
                operation_type=OperationType.ADD,
                message=f"成功添加知识项: {item.title}",
                item_id=item.id,
                data={"item": item.to_dict()}
            )
        except Exception as e:
            return OperationResult(
                success=False,
                operation_type=OperationType.ADD,
                message=f"添加失败: {str(e)}"
            )
    
    def execute_merge(self, op: MergeOperation) -> OperationResult:
        """
        执行合并操作
        
        参数:
            op: 合并操作对象
        
        返回:
            操作结果
        """
        # 验证参数
        valid, error = op.validate()
        if not valid:
            return OperationResult(
                success=False,
                operation_type=OperationType.MERGE,
                message=f"参数验证失败: {error}"
            )
        
        try:
            # 检查源项是否存在
            source_items = []
            for item_id in op.source_ids:
                item = self.graph.get_item(item_id)
                if item is None:
                    return OperationResult(
                        success=False,
                        operation_type=OperationType.MERGE,
                        message=f"源知识项不存在: {item_id}"
                    )
                source_items.append(item)
            
            # 创建合并后的知识项
            merged_item = GlobalGraphItem(
                title=op.merged_title.strip(),
                content=op.merged_content.strip(),
            )
            
            # 添加合并后的项
            self.graph.add_item(merged_item)
            
            # 删除源项 (如果指定)
            deleted_ids = []
            if op.delete_sources:
                for item_id in op.source_ids:
                    if self.graph.remove_item(item_id):
                        deleted_ids.append(item_id)
            
            return OperationResult(
                success=True,
                operation_type=OperationType.MERGE,
                message=f"成功合并 {len(source_items)} 个知识项",
                item_id=merged_item.id,
                data={
                    "merged_item": merged_item.to_dict(),
                    "source_ids": op.source_ids,
                    "deleted_ids": deleted_ids,
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                operation_type=OperationType.MERGE,
                message=f"合并失败: {str(e)}"
            )
    
    def execute_delete(self, op: DeleteOperation) -> OperationResult:
        """
        执行删除操作
        
        参数:
            op: 删除操作对象
        
        返回:
            操作结果
        """
        # 验证参数
        valid, error = op.validate()
        if not valid:
            return OperationResult(
                success=False,
                operation_type=OperationType.DELETE,
                message=f"参数验证失败: {error}"
            )
        
        try:
            # 获取要删除的项 (用于日志)
            item = self.graph.get_item(op.item_id)
            if item is None:
                return OperationResult(
                    success=False,
                    operation_type=OperationType.DELETE,
                    message=f"知识项不存在: {op.item_id}"
                )
            
            item_title = item.title
            
            # 执行删除
            if self.graph.remove_item(op.item_id):
                return OperationResult(
                    success=True,
                    operation_type=OperationType.DELETE,
                    message=f"成功删除知识项: {item_title}",
                    item_id=op.item_id,
                    data={"reason": op.reason}
                )
            else:
                return OperationResult(
                    success=False,
                    operation_type=OperationType.DELETE,
                    message=f"删除失败: {op.item_id}"
                )
        except Exception as e:
            return OperationResult(
                success=False,
                operation_type=OperationType.DELETE,
                message=f"删除失败: {str(e)}"
            )
    
    def execute_modify(self, op: ModifyOperation) -> OperationResult:
        """
        执行修改操作
        
        参数:
            op: 修改操作对象
        
        返回:
            操作结果
        """
        # 验证参数
        valid, error = op.validate()
        if not valid:
            return OperationResult(
                success=False,
                operation_type=OperationType.MODIFY,
                message=f"参数验证失败: {error}"
            )
        
        try:
            # 获取要修改的项
            item = self.graph.get_item(op.item_id)
            if item is None:
                return OperationResult(
                    success=False,
                    operation_type=OperationType.MODIFY,
                    message=f"知识项不存在: {op.item_id}"
                )
            
            # 记录修改前的状态
            old_state = item.to_dict()
            
            # 执行修改
            updates = {}
            if op.new_title is not None:
                updates["title"] = op.new_title.strip()
            if op.new_content is not None:
                updates["content"] = op.new_content.strip()
            
            item.update(**updates)
            
            return OperationResult(
                success=True,
                operation_type=OperationType.MODIFY,
                message=f"成功修改知识项: {item.title}",
                item_id=op.item_id,
                data={
                    "old_state": old_state,
                    "new_state": item.to_dict(),
                    "updated_fields": list(updates.keys()),
                }
            )
        except Exception as e:
            return OperationResult(
                success=False,
                operation_type=OperationType.MODIFY,
                message=f"修改失败: {str(e)}"
            )
    
    # ==================== LLM 响应解析 ====================
    
    def parse_llm_response(self, response: str) -> List[Any]:
        """
        解析 LLM 响应为操作列表
        
        支持的响应格式:
            1. JSON 格式: {"operations": [...]}
            2. JSON 数组: [...]
            3. 单个操作 JSON: {...}
        
        参数:
            response: LLM 的原始响应文本
        
        返回:
            操作对象列表 (AddOperation, MergeOperation, etc.)
        """
        operations = []
        
        # 尝试提取 JSON
        json_data = self._extract_json(response)
        if json_data is None:
            return operations
        
        # 处理不同格式
        if isinstance(json_data, dict):
            if "operations" in json_data:
                op_list = json_data["operations"]
            else:
                op_list = [json_data]
        elif isinstance(json_data, list):
            op_list = json_data
        else:
            return operations
        
        # 解析每个操作
        for op_data in op_list:
            op = self._parse_single_operation(op_data)
            if op is not None:
                operations.append(op)
        
        return operations
    
    def _extract_json(self, text: str) -> Optional[Any]:
        """
        从文本中提取 JSON
        
        参数:
            text: 原始文本
        
        返回:
            解析后的 JSON 对象，解析失败返回 None
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 代码块
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 尝试提取花括号/方括号内容
        brace_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(brace_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _parse_single_operation(self, data: Dict[str, Any]) -> Optional[Any]:
        """
        解析单个操作
        
        参数:
            data: 操作数据字典
        
        返回:
            操作对象，解析失败返回 None
        """
        op_type = data.get("operation", "").lower()
        
        if op_type == "add":
            return AddOperation(
                title=data.get("title", ""),
                content=data.get("content", ""),
            )
        
        elif op_type == "merge":
            return MergeOperation(
                source_ids=data.get("source_ids", []),
                merged_title=data.get("merged_title", ""),
                merged_content=data.get("merged_content", ""),
                delete_sources=data.get("delete_sources", True),
            )
        
        elif op_type == "delete":
            return DeleteOperation(
                item_id=data.get("item_id", ""),
                reason=data.get("reason"),
            )
        
        elif op_type == "modify":
            return ModifyOperation(
                item_id=data.get("item_id", ""),
                new_title=data.get("new_title"),
                new_content=data.get("new_content"),
            )
        
        return None
    
    # ==================== 批量执行 ====================
    
    def execute_operations(
        self,
        operations: List[Any],
        stop_on_error: bool = False,
    ) -> List[OperationResult]:
        """
        批量执行操作
        
        参数:
            operations: 操作对象列表
            stop_on_error: 遇到错误是否停止执行
        
        返回:
            操作结果列表
        """
        results = []
        
        for op in operations:
            if isinstance(op, AddOperation):
                result = self.execute_add(op)
            elif isinstance(op, MergeOperation):
                result = self.execute_merge(op)
            elif isinstance(op, DeleteOperation):
                result = self.execute_delete(op)
            elif isinstance(op, ModifyOperation):
                result = self.execute_modify(op)
            else:
                result = OperationResult(
                    success=False,
                    operation_type=OperationType.ADD,  # 默认类型
                    message=f"未知的操作类型: {type(op).__name__}"
                )
            
            results.append(result)
            
            if stop_on_error and not result.success:
                break
        
        return results
    
    # ==================== 便捷方法 ====================
    
    def add_item(
        self,
        title: str,
        content: str,
    ) -> OperationResult:
        """
        便捷方法: 添加知识项
        
        参数:
            title: 标题
            content: 内容
        返回:
            操作结果
        """
        op = AddOperation(
            title=title,
            content=content,
        )
        return self.execute_add(op)
    
    def delete_item(self, item_id: str, reason: str = None) -> OperationResult:
        """
        便捷方法: 删除知识项
        
        参数:
            item_id: 知识项 ID
            reason: 删除原因
        
        返回:
            操作结果
        """
        op = DeleteOperation(item_id=item_id, reason=reason)
        return self.execute_delete(op)
    
    def modify_item(
        self,
        item_id: str,
        title: str = None,
        content: str = None,
    ) -> OperationResult:
        """
        便捷方法: 修改知识项
        
        参数:
            item_id: 知识项 ID
            title: 新标题 (None 表示不修改)
            content: 新内容 (None 表示不修改)
        返回:
            操作结果
        """
        op = ModifyOperation(
            item_id=item_id,
            new_title=title,
            new_content=content,
        )
        return self.execute_modify(op)
    
    def merge_items(
        self,
        source_ids: List[str],
        merged_title: str,
        merged_content: str,
        delete_sources: bool = True,
    ) -> OperationResult:
        """
        便捷方法: 合并知识项
        
        参数:
            source_ids: 源知识项 ID 列表
            merged_title: 合并后的标题
            merged_content: 合并后的内容
            delete_sources: 是否删除源项
        
        返回:
            操作结果
        """
        op = MergeOperation(
            source_ids=source_ids,
            merged_title=merged_title,
            merged_content=merged_content,
            delete_sources=delete_sources,
        )
        return self.execute_merge(op)
