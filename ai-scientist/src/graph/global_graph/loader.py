"""
Global Graph 加载器

负责全局图的加载、保存和初始化。

存储路径: data/{domain}/{task}/global_graph.json

作者: GraphScientist Team
版本: 1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from graph.global_graph.data_structures import GlobalGraph, GlobalGraphItem


class GlobalGraphLoader:
    """
    全局图加载器
    
    负责从文件系统加载和保存全局图。
    
    属性:
        data_root: 数据根目录
    
    存储路径格式:
        {data_root}/{domain}/{task}/global_graph.json
    
    使用示例:
        >>> loader = GlobalGraphLoader(data_root="data")
        >>> graph = loader.load("Recsys", "MultiModal")
        >>> if graph is None:
        ...     graph = loader.create_default("Recsys", "MultiModal")
        >>> loader.save(graph, "Recsys", "MultiModal")
    """
    
    # 默认文件名
    DEFAULT_FILENAME = "global_graph.json"
    
    def __init__(self, data_root: str = "data"):
        """
        初始化加载器
        
        参数:
            data_root: 数据根目录路径
        """
        self.data_root = Path(data_root)
    
    def _get_file_path(self, domain: str, task: str) -> Path:
        """
        获取全局图文件路径
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            文件路径
        """
        return self.data_root / domain / task / self.DEFAULT_FILENAME
    
    def load(self, domain: str, task: str) -> Optional[GlobalGraph]:
        """
        加载全局图
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            GlobalGraph 实例，如果文件不存在则返回 None
        """
        file_path = self._get_file_path(domain, task)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return GlobalGraph.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[GlobalGraphLoader] 加载失败: {e}")
            return None
    
    def save(self, graph: GlobalGraph, domain: str, task: str) -> bool:
        """
        保存全局图
        
        参数:
            graph: GlobalGraph 实例
            domain: 领域名称
            task: 任务名称
        
        返回:
            是否保存成功
        """
        file_path = self._get_file_path(domain, task)
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = graph.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[GlobalGraphLoader] 保存失败: {e}")
            return False
    
    def exists(self, domain: str, task: str) -> bool:
        """
        检查全局图文件是否存在
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            文件是否存在
        """
        return self._get_file_path(domain, task).exists()
    
    def delete(self, domain: str, task: str) -> bool:
        """
        删除全局图文件
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            是否删除成功
        """
        file_path = self._get_file_path(domain, task)
        
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"[GlobalGraphLoader] 删除失败: {e}")
                return False
        return False
    
    def create_default(self, domain: str, task: str) -> GlobalGraph:
        """
        创建默认的全局图
        
        根据领域和任务创建带有默认知识项的全局图。
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            新创建的 GlobalGraph 实例
        """
        graph = GlobalGraph(domain=domain, task=task)
        
        # 根据领域添加默认知识项
        default_items = self._get_default_items(domain, task)
        for item in default_items:
            graph.add_item(item)
        
        return graph
    
    def load_or_create(self, domain: str, task: str) -> GlobalGraph:
        """
        加载全局图，如果不存在则创建默认的
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            GlobalGraph 实例
        """
        graph = self.load(domain, task)
        if graph is None:
            graph = self.create_default(domain, task)
            self.save(graph, domain, task)
        return graph
    
    def _get_default_items(self, domain: str, task: str) -> list[GlobalGraphItem]:
        """
        获取默认知识项
        
        参数:
            domain: 领域名称
            task: 任务名称
        
        返回:
            默认知识项列表
        """
        # 推荐系统 - 多模态推荐
        if domain == "Recsys" and task in ["MultiModal", "MultiModalRecommendation"]:
            return [
                GlobalGraphItem(
                    title="多模态特征融合策略",
                    content="""多模态推荐系统中常见的特征融合策略包括:

1. **早期融合 (Early Fusion)**: 在输入层将不同模态的特征拼接
   - 优点: 简单直接
   - 缺点: 无法捕捉模态间的交互

2. **晚期融合 (Late Fusion)**: 各模态独立编码后在决策层融合
   - 优点: 保留模态特有信息
   - 缺点: 计算量较大

3. **注意力融合 (Attention Fusion)**: 使用注意力机制动态融合
   - 优点: 能够自适应地分配权重
   - 缺点: 增加模型复杂度

4. **对比学习融合 (Contrastive Fusion)**: 通过对比学习对齐模态空间
   - 优点: 能学习到更好的表示
   - 缺点: 需要设计合适的对比目标""",
                ),
                GlobalGraphItem(
                    title="BPR 损失函数",
                    content="""BPR (Bayesian Personalized Ranking) 是推荐系统中最常用的损失函数之一:

```python
def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
```

核心思想: 正样本的得分应该高于负样本的得分。

变体:
- Weighted BPR: 对不同负样本赋予不同权重
- Sampled Softmax: 使用 softmax 形式的损失""",
                ),
                GlobalGraphItem(
                    title="图神经网络在推荐中的应用",
                    content="""图神经网络 (GNN) 在推荐系统中的典型应用:

1. **用户-物品二部图**: 建模用户和物品的交互关系
2. **LightGCN**: 简化的图卷积，去除特征变换和非线性
3. **消息传递**: 聚合邻居信息更新节点表示

关键代码模式:
```python
# LightGCN 风格的邻居聚合
def aggregate(self, edge_index, x):
    row, col = edge_index
    deg = degree(col, x.size(0))
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return scatter_add(x[row] * norm.view(-1, 1), col, dim=0)
```""",
                ),
                GlobalGraphItem(
                    title="多模态推荐评估指标",
                    content="""常用的推荐系统评估指标:

1. **Recall@K**: 在 Top-K 推荐中召回了多少正样本
2. **NDCG@K**: 考虑排序位置的增益指标
3. **Hit@K**: Top-K 中是否包含正样本
4. **MRR**: 第一个正样本的排名倒数

计算示例:
```python
def recall_at_k(scores, labels, k):
    topk = torch.topk(scores, k).indices
    hits = torch.gather(labels, 1, topk).sum(1)
    return (hits / labels.sum(1).clamp(min=1)).mean()
```""",
                ),
            ]
        
        # 默认返回空列表
        return []
