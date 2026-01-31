"""
Global Graph 模块

全局领域知识图谱，用于存储和管理跨论文的领域知识。

模块结构:
    - data_structures.py: 数据结构定义 (GlobalGraphItem, GlobalGraph)
    - loader.py: 加载和保存 (GlobalGraphLoader)
    - llm_operations.py: LLM 驱动的图操作 (添加、删除、修改、合并知识项)

数据存储:
    全局图存储在 data/{domain}/{task}/global_graph.json

使用示例:
    # 基础使用
    from graph.global_graph import GlobalGraph, GlobalGraphItem, GlobalGraphLoader
    
    loader = GlobalGraphLoader(data_root="data")
    graph = loader.load_or_create("Recsys", "MultiModal")
    
    # LLM 驱动的操作
    from graph.global_graph import GlobalGraphLLMOperator
    
    operator = GlobalGraphLLMOperator(graph)
    
    # 添加知识项
    result = operator.add_item(
        title="多模态特征融合",
        content="..."
    )
    
    # 解析 LLM 响应并执行操作
    operations = operator.parse_llm_response(llm_output)
    results = operator.execute_operations(operations)

作者: GraphScientist Team
版本: 1.0.0
"""

from graph.global_graph.data_structures import GlobalGraphItem, GlobalGraph
from graph.global_graph.loader import GlobalGraphLoader
from graph.global_graph.llm_operations import (
    # 操作类型
    OperationType,
    OperationResult,
    # 操作对象
    AddOperation,
    MergeOperation,
    DeleteOperation,
    ModifyOperation,
    # LLM 操作器
    GlobalGraphLLMOperator,
)

__all__ = [
    # 数据结构
    "GlobalGraphItem",
    "GlobalGraph",
    # 加载器
    "GlobalGraphLoader",
    # LLM 操作类型
    "OperationType",
    "OperationResult",
    # LLM 操作对象
    "AddOperation",
    "MergeOperation",
    "DeleteOperation",
    "ModifyOperation",
    # LLM 操作器
    "GlobalGraphLLMOperator",
]
