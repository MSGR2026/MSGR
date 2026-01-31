"""
生成语义边信息的程序，
接收当前节点（边的终点）的方法部分，接收相关节点（边的起点）的方法部分和示例代码，加载提示词模板（在/home/lilehui/rnn/ai-scientist/ai-scientist/src/agent/prompt/Recsys/MultiModal/semantic_edge.jinja中定义），拼接出提示词，
要求大模型输出语义边信息，包括两个节点的相同之处，不同之处，并填充到对应的文件中/home/lilehui/rnn/ai-scientist/ai-scientist/data/Recsys/MultiModalRecommendation/local/edge.json
大模型同时还要识别相关节点的方法和示例代码的差别，因为很多时候示例代码相比于原始论文会引入一些工程优化。如果两个节点都用了某个组件，且相关节点中对该组件做了工程优化，那么大模型应该在语义边中显式的指出这一点

# 为指定论文生成所有以它为 target 的边（最常用）
python semantic_edge_generator.py --domain TimeSeries --task long_term_forecast \
    --paper Autoformer_2021 --as-target

# 为指定论文生成所有以它为 source 的边
python semantic_edge_generator.py --domain TimeSeries --task long_term_forecast \
    --paper Informer_2020 --as-source

# 为指定论文生成所有相关的边（source + target）
python semantic_edge_generator.py --domain TimeSeries --task long_term_forecast \
    --paper PatchTST_2022

# 强制重新生成（即使已有语义信息）
python semantic_edge_generator.py --domain TimeSeries --task long_term_forecast \
    --paper Autoformer_2021 --as-target --force

# 生成所有缺少语义信息的边
python semantic_edge_generator.py --domain TimeSeries --task long_term_forecast \
    --all-missing

SemanticEdgeGenerator - 语义边信息生成器

本模块负责生成论文节点之间的语义边信息。

功能特性:
    1. 接收当前节点（边的终点）的方法部分
    2. 接收相关节点（边的起点）的方法部分和示例代码
    3. 加载提示词模板并拼接提示词
    4. 调用大模型生成语义边信息
    5. 识别相同之处、不同之处
    6. 识别示例代码相比于原始论文的工程优化
    7. 将结果填充到 edge.json 文件中

作者: GraphScientist Team
版本: 1.0.0
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# 添加 src 目录到路径
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from storage.domain_task_storage import DomainTaskStorage, GraphEdge

if TYPE_CHECKING:
    from llm.client import LLMClient
    from graph.local_graph.local_graph_loader import LocalGraphLoader


@dataclass
class SemanticEdgeInfo:
    """语义边信息数据结构
    
    Attributes:
        source: 边的起点（相关论文/先前工作）
        target: 边的终点（当前论文/后续工作）
        edge_type: 边类型 ("in-domain" 或 "out-of-domain")
        similarities: 两个节点的相同之处（包含工程优化说明）
        differences: 两个节点的不同之处
    """
    source: str
    target: str
    edge_type: str = "in-domain"
    similarities: Optional[str] = None
    differences: Optional[str] = None
    
    def to_graph_edge(self) -> GraphEdge:
        """转换为 GraphEdge 对象
        
        注意：新创建的边没有 rank 信息，rank 字段为 None。
        如果需要保留原有边的 rank，应该使用 _save_edge 方法的更新逻辑。
        """
        return GraphEdge(
            source=self.source,
            target=self.target,
            type=self.edge_type,
            similarities=self.similarities,
            differences=self.differences,
            relation=None,
            rank=None,  # 新边没有 rank 信息
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "similarities": self.similarities,
            "differences": self.differences,
        }


class SemanticEdgeGenerator:
    """
    语义边信息生成器
    
    负责分析两个论文节点之间的关系，生成语义边信息，包括：
    - 方法论上的相同点
    - 方法论上的不同点
    - 示例代码中的工程优化
    
    Attributes:
        llm_client: LLM 客户端
        loader: LocalGraphLoader 实例
        domain: 领域名称
        task: 任务名称
        storage: 存储管理器
    
    使用示例:
        >>> generator = SemanticEdgeGenerator(
        ...     llm_client=client,
        ...     local_graph_loader=loader,
        ...     domain="Recsys",
        ...     task="MultiModalRecommendation"
        ... )
        >>> edge_info = generator.generate_edge(
        ...     source_paper_id="DualGNN_2021",
        ...     target_paper_id="BM3_2023"
        ... )
    """
    
    # 模板目录
    TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "agent" / "prompt"
    
    # 任务名称映射
    TASK_MAPPING = {
        # Recsys 领域
        "MultiModalRecommendation": "MultiModal",
        "MultiModal": "MultiModal",
        # TimeSeries 领域
        "long_term_forecast": "long_term_forecast",
        "short_term_forecast": "short_term_forecast",
        "imputation": "imputation",
        "anomaly_detection": "anomaly_detection",
        "classification": "classification",
    }
    
    def __init__(
        self,
        llm_client: "LLMClient",
        local_graph_loader: "LocalGraphLoader",
        domain: str = "Recsys",
        task: str = "MultiModalRecommendation",
        data_root: Optional[Path] = None,
    ):
        """
        初始化语义边生成器
        
        Args:
            llm_client: LLM 客户端实例
            local_graph_loader: LocalGraphLoader 实例
            domain: 领域名称
            task: 任务名称
            data_root: 数据根目录路径
        """
        self.llm = llm_client
        self.loader = local_graph_loader
        self.domain = domain
        self.task = task
        self.task_dir = self.TASK_MAPPING.get(task, task)
        
        # 初始化存储
        if data_root is None:
            data_root = self.loader.data_root
        self.storage = DomainTaskStorage(data_root)
        
        # 初始化 Jinja2 环境
        self._init_jinja_env()
    
    def _init_jinja_env(self) -> None:
        """初始化 Jinja2 模板环境"""
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
        except ImportError:
            raise ImportError("请安装 jinja2: pip install jinja2")
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.TEMPLATE_DIR)),
            autoescape=select_autoescape(disabled_extensions=('jinja',)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    
    def _get_template_path(self) -> str:
        """获取语义边模板路径"""
        return f"{self.domain}/{self.task_dir}/semantic_edge.jinja"
    
    def _load_template(self) -> Any:
        """加载 Jinja2 模板"""
        template_path = self._get_template_path()
        return self.jinja_env.get_template(template_path)
    
    def _collect_context(
        self,
        source_paper_id: str,
        target_paper_id: str,
    ) -> Dict[str, Any]:
        """
        收集生成语义边所需的上下文数据
        
        Args:
            source_paper_id: 边起点论文ID（先前工作）
            target_paper_id: 边终点论文ID（后续工作）
        
        Returns:
            包含所有模板变量的字典
        """
        # 获取源节点（先前工作）信息
        source_method = self.loader.get_method(source_paper_id)
        source_introduction = self.loader.get_introduction(source_paper_id)
        source_algorithm = self.loader.get_algorithm_implementation(source_paper_id)
        source_info = self.loader.get_paper_info(source_paper_id)
        
        # 获取目标节点（后续工作）信息
        target_method = self.loader.get_method(target_paper_id)
        target_introduction = self.loader.get_introduction(target_paper_id)
        target_algorithm = self.loader.get_algorithm_implementation(target_paper_id)
        target_info = self.loader.get_paper_info(target_paper_id)
        
        return {
            "domain": self.domain,
            "task": self.task,
            # 源节点信息（先前工作）
            "source_paper_id": source_paper_id,
            "source_paper_title": source_info.get("paper_title", "") if source_info else "",
            "source_alias": source_info.get("alias", "") if source_info else "",
            "source_year": source_info.get("year", 0) if source_info else 0,
            "source_method": source_method,
            "source_introduction": source_introduction,
            "source_algorithm_code": source_algorithm,
            # 目标节点信息（后续工作）
            "target_paper_id": target_paper_id,
            "target_paper_title": target_info.get("paper_title", "") if target_info else "",
            "target_alias": target_info.get("alias", "") if target_info else "",
            "target_year": target_info.get("year", 0) if target_info else 0,
            "target_method": target_method,
            "target_introduction": target_introduction,
            "target_algorithm_code": target_algorithm,
        }
    
    def _render_prompt(self, context: Dict[str, Any]) -> tuple[str, str]:
        """
        渲染提示词模板
        
        Args:
            context: 上下文数据
        
        Returns:
            (system_prompt, user_prompt) 元组
        """
        template = self._load_template()
        rendered = template.render(**context)
        
        # 解析分隔符
        SYSTEM_MARKER = "---SYSTEM_PROMPT---"
        USER_MARKER = "---USER_PROMPT---"
        
        if SYSTEM_MARKER not in rendered or USER_MARKER not in rendered:
            raise ValueError(
                f"模板格式错误: 必须同时包含 {SYSTEM_MARKER} 和 {USER_MARKER} 分隔符"
            )
        
        parts = rendered.split(USER_MARKER)
        user_prompt = parts[1].strip() if len(parts) > 1 else ""
        system_part = parts[0]
        system_prompt = system_part.replace(SYSTEM_MARKER, "").strip()
        
        return system_prompt, user_prompt
    
    def _parse_llm_response(self, response: str) -> SemanticEdgeInfo:
        """
        解析 LLM 响应，提取语义边信息
        
        Args:
            response: LLM 响应文本
        
        Returns:
            SemanticEdgeInfo 对象
        """
        # 尝试从 JSON 块中提取
        import re
        
        # 查找 JSON 代码块
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return SemanticEdgeInfo(
                    source=data.get("source", ""),
                    target=data.get("target", ""),
                    edge_type=data.get("type", "in-domain"),
                    similarities=data.get("similarities"),
                    differences=data.get("differences"),
                )
            except json.JSONDecodeError:
                pass
        
        # 如果没有 JSON，尝试从文本中解析
        similarities = None
        differences = None
        
        # 解析相同点
        sim_pattern = r'(?:相同[之点处]|Similarities?)[：:]\s*([\s\S]*?)(?=(?:不同[之点处]|Differences?)[：:]|$)'
        sim_match = re.search(sim_pattern, response, re.IGNORECASE)
        if sim_match:
            similarities = sim_match.group(1).strip()
        
        # 解析不同点
        diff_pattern = r'(?:不同[之点处]|Differences?)[：:]\s*([\s\S]*?)$'
        diff_match = re.search(diff_pattern, response, re.IGNORECASE)
        if diff_match:
            differences = diff_match.group(1).strip()
        
        return SemanticEdgeInfo(
            source="",  # 需要在调用处设置
            target="",  # 需要在调用处设置
            edge_type="in-domain",
            similarities=similarities,
            differences=differences,
        )
    
    def generate_edge(
        self,
        source_paper_id: str,
        target_paper_id: str,
        save_to_file: bool = True,
    ) -> SemanticEdgeInfo:
        """
        生成两个论文之间的语义边信息
        
        Args:
            source_paper_id: 边起点论文ID（先前工作）
            target_paper_id: 边终点论文ID（后续工作）
            save_to_file: 是否保存到 edge.json 文件
        
        Returns:
            SemanticEdgeInfo 对象
        """
        print(f"[SemanticEdgeGenerator] 生成语义边: {source_paper_id} -> {target_paper_id}")
        
        # 收集上下文
        context = self._collect_context(source_paper_id, target_paper_id)
        
        # 渲染提示词
        system_prompt, user_prompt = self._render_prompt(context)
        
        print(f"[SemanticEdgeGenerator] 提示词长度: system={len(system_prompt)}, user={len(user_prompt)}")
        
        # 调用 LLM
        response = self.llm.generate(system_prompt, user_prompt)
        
        # 解析响应
        edge_info = self._parse_llm_response(response)
        edge_info.source = source_paper_id
        edge_info.target = target_paper_id
        
        print(f"[SemanticEdgeGenerator] 生成完成:")
        print(f"  - 相同点: {len(edge_info.similarities or '')} 字符")
        print(f"  - 不同点: {len(edge_info.differences or '')} 字符")
        
        # 保存到文件
        if save_to_file:
            self._save_edge(edge_info)
        
        return edge_info
    
    def _save_edge(self, edge_info: SemanticEdgeInfo) -> None:
        """
        保存语义边到 edge.json 文件
        
        如果边已存在，则更新语义信息（保留原有的 type 等字段）；否则添加新边。
        
        Args:
            edge_info: 语义边信息
        """
        # 获取现有边
        edges = self.storage.get_edges(self.domain, self.task)
        
        # 查找是否已存在该边
        existing_idx = None
        existing_edge = None
        for i, edge in enumerate(edges):
            if edge.source == edge_info.source and edge.target == edge_info.target:
                existing_idx = i
                existing_edge = edge
                break
        
        if existing_idx is not None:
            # 更新现有边：只更新语义信息，保留原有的 type 等字段
            existing_edge.similarities = edge_info.similarities
            existing_edge.differences = edge_info.differences
            # 注意：保留原有的 type 字段，不覆盖
            print(f"[SemanticEdgeGenerator] 更新边语义: {edge_info.source} -> {edge_info.target}")
        else:
            # 添加新边
            new_edge = edge_info.to_graph_edge()
            edges.append(new_edge)
            print(f"[SemanticEdgeGenerator] 添加新边: {edge_info.source} -> {edge_info.target}")
        
        # 保存
        file_path = self.storage.save_edges(self.domain, self.task, edges)
        print(f"[SemanticEdgeGenerator] 已保存到: {file_path}")
    
    def generate_edges_batch(
        self,
        edge_pairs: List[tuple[str, str]],
        skip_existing: bool = True,
    ) -> List[SemanticEdgeInfo]:
        """
        批量生成语义边
        
        Args:
            edge_pairs: 边列表，每个元素为 (source_paper_id, target_paper_id)
            skip_existing: 是否跳过已存在的边
        
        Returns:
            生成的 SemanticEdgeInfo 列表
        """
        results = []
        
        # 获取现有边
        existing_edges = set()
        if skip_existing:
            edges = self.storage.get_edges(self.domain, self.task)
            for edge in edges:
                # 只跳过已有完整语义信息的边
                if edge.similarities or edge.differences:
                    existing_edges.add((edge.source, edge.target))
        
        for source_id, target_id in edge_pairs:
            if skip_existing and (source_id, target_id) in existing_edges:
                print(f"[SemanticEdgeGenerator] 跳过已存在的边: {source_id} -> {target_id}")
                continue
            
            try:
                edge_info = self.generate_edge(source_id, target_id)
                results.append(edge_info)
            except Exception as e:
                print(f"[SemanticEdgeGenerator] 生成失败 ({source_id} -> {target_id}): {e}")
        
        return results
    
    def generate_all_missing_edges(self) -> List[SemanticEdgeInfo]:
        """
        为所有缺少语义信息的边生成语义边
        
        Returns:
            生成的 SemanticEdgeInfo 列表
        """
        edges = self.storage.get_edges(self.domain, self.task)
        
        # 找出缺少语义信息的边
        missing_pairs = []
        for edge in edges:
            if not edge.similarities and not edge.differences:
                missing_pairs.append((edge.source, edge.target))
        
        print(f"[SemanticEdgeGenerator] 发现 {len(missing_pairs)} 条缺少语义信息的边")
        
        return self.generate_edges_batch(missing_pairs, skip_existing=False)
    
    def generate_all_edges(
        self,
        skip_existing: bool = True,
    ) -> List[SemanticEdgeInfo]:
        """
        为领域任务中的所有边生成语义信息
        
        Args:
            skip_existing: 是否跳过已有语义信息的边
        
        Returns:
            生成的 SemanticEdgeInfo 列表
        """
        edges = self.storage.get_edges(self.domain, self.task)
        
        # 收集所有边
        all_pairs = []
        skipped_count = 0
        for edge in edges:
            if skip_existing and (edge.similarities or edge.differences):
                skipped_count += 1
                continue
            all_pairs.append((edge.source, edge.target))
        
        total_edges = len(edges)
        to_generate = len(all_pairs)
        
        print(f"[SemanticEdgeGenerator] 领域: {self.domain}, 任务: {self.task}")
        print(f"[SemanticEdgeGenerator] 总边数: {total_edges}, 待生成: {to_generate}, 跳过: {skipped_count}")
        
        if not all_pairs:
            print(f"[SemanticEdgeGenerator] 没有需要生成的边")
            return []
        
        return self.generate_edges_batch(all_pairs, skip_existing=False)
    
    def generate_edges_for_target(
        self,
        target_paper_id: str,
        skip_existing: bool = True,
    ) -> List[SemanticEdgeInfo]:
        """
        为指定论文生成所有以它为 target 的语义边
        
        Args:
            target_paper_id: 目标论文ID
            skip_existing: 是否跳过已有语义信息的边
        
        Returns:
            生成的 SemanticEdgeInfo 列表
        """
        edges = self.storage.get_edges(self.domain, self.task)
        
        # 找出所有以该论文为 target 的边
        target_pairs = []
        for edge in edges:
            if edge.target == target_paper_id:
                # 根据 skip_existing 决定是否跳过
                if skip_existing and (edge.similarities or edge.differences):
                    print(f"[SemanticEdgeGenerator] 跳过已有语义信息的边: {edge.source} -> {edge.target}")
                    continue
                target_pairs.append((edge.source, edge.target))
        
        if not target_pairs:
            print(f"[SemanticEdgeGenerator] 未找到以 {target_paper_id} 为 target 的边")
            return []
        
        print(f"[SemanticEdgeGenerator] 找到 {len(target_pairs)} 条以 {target_paper_id} 为 target 的边")
        
        return self.generate_edges_batch(target_pairs, skip_existing=False)
    
    def generate_edges_for_source(
        self,
        source_paper_id: str,
        skip_existing: bool = True,
    ) -> List[SemanticEdgeInfo]:
        """
        为指定论文生成所有以它为 source 的语义边
        
        Args:
            source_paper_id: 源论文ID
            skip_existing: 是否跳过已有语义信息的边
        
        Returns:
            生成的 SemanticEdgeInfo 列表
        """
        edges = self.storage.get_edges(self.domain, self.task)
        
        # 找出所有以该论文为 source 的边
        source_pairs = []
        for edge in edges:
            if edge.source == source_paper_id:
                # 根据 skip_existing 决定是否跳过
                if skip_existing and (edge.similarities or edge.differences):
                    print(f"[SemanticEdgeGenerator] 跳过已有语义信息的边: {edge.source} -> {edge.target}")
                    continue
                source_pairs.append((edge.source, edge.target))
        
        if not source_pairs:
            print(f"[SemanticEdgeGenerator] 未找到以 {source_paper_id} 为 source 的边")
            return []
        
        print(f"[SemanticEdgeGenerator] 找到 {len(source_pairs)} 条以 {source_paper_id} 为 source 的边")
        
        return self.generate_edges_batch(source_pairs, skip_existing=False)
    
    def generate_edges_for_paper(
        self,
        paper_id: str,
        skip_existing: bool = True,
    ) -> List[SemanticEdgeInfo]:
        """
        为指定论文生成所有相关的语义边（包括作为 source 和 target）
        
        Args:
            paper_id: 论文ID
            skip_existing: 是否跳过已有语义信息的边
        
        Returns:
            生成的 SemanticEdgeInfo 列表
        """
        edges = self.storage.get_edges(self.domain, self.task)
        
        # 找出所有与该论文相关的边
        related_pairs = []
        for edge in edges:
            if edge.source == paper_id or edge.target == paper_id:
                # 根据 skip_existing 决定是否跳过
                if skip_existing and (edge.similarities or edge.differences):
                    print(f"[SemanticEdgeGenerator] 跳过已有语义信息的边: {edge.source} -> {edge.target}")
                    continue
                related_pairs.append((edge.source, edge.target))
        
        if not related_pairs:
            print(f"[SemanticEdgeGenerator] 未找到与 {paper_id} 相关的边")
            return []
        
        print(f"[SemanticEdgeGenerator] 找到 {len(related_pairs)} 条与 {paper_id} 相关的边")
        
        return self.generate_edges_batch(related_pairs, skip_existing=False)


# ==================== 命令行入口 ====================

def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成论文之间的语义边信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成单条边
  python semantic_edge_generator.py --source Informer_2020 --target Autoformer_2021

  # 为指定论文生成所有以它为 target 的边
  python semantic_edge_generator.py --paper Autoformer_2021 --as-target

  # 为指定论文生成所有以它为 source 的边
  python semantic_edge_generator.py --paper Informer_2020 --as-source

  # 为指定论文生成所有相关的边（source 和 target）
  python semantic_edge_generator.py --paper PatchTST_2022

  # 为所有缺少语义信息的边生成
  python semantic_edge_generator.py --all-missing

  # 为领域任务中的所有边生成语义信息
  python semantic_edge_generator.py --domain TimeSeries --task anomaly_detection --all

  # 强制重新生成所有边（包括已有语义信息的）
  python semantic_edge_generator.py --domain TimeSeries --task anomaly_detection --all --force

  # 指定时序领域
  python semantic_edge_generator.py --domain TimeSeries --task long_term_forecast --paper Autoformer_2021 --as-target
        """
    )
    parser.add_argument("--source", type=str, help="源论文ID（先前工作）")
    parser.add_argument("--target", type=str, help="目标论文ID（后续工作）")
    parser.add_argument("--paper", type=str, help="论文ID，用于批量生成相关的边")
    parser.add_argument("--as-target", action="store_true", help="与 --paper 配合使用，只生成以该论文为 target 的边")
    parser.add_argument("--as-source", action="store_true", help="与 --paper 配合使用，只生成以该论文为 source 的边")
    parser.add_argument("--domain", type=str, default="Recsys", help="领域名称 (默认: Recsys)")
    parser.add_argument("--task", type=str, default="MultiModalRecommendation", help="任务名称 (默认: MultiModalRecommendation)")
    parser.add_argument("--all", action="store_true", help="为领域任务中的所有边生成语义信息")
    parser.add_argument("--all-missing", action="store_true", help="为所有缺少语义信息的边生成")
    parser.add_argument("--force", action="store_true", help="强制重新生成，即使已有语义信息")
    parser.add_argument("--config", type=str, default="configs/llm.yaml", help="LLM 配置文件路径")
    
    args = parser.parse_args()
    
    # 导入必要模块
    from llm.client import LLMClient
    from graph.local_graph.local_graph_loader import LocalGraphLoader
    
    # 初始化
    llm_client = LLMClient(config_path=args.config)
    loader = LocalGraphLoader(domain=args.domain, task=args.task)
    generator = SemanticEdgeGenerator(
        llm_client=llm_client,
        local_graph_loader=loader,
        domain=args.domain,
        task=args.task,
    )
    
    skip_existing = not args.force
    
    if args.all:
        # 为领域任务中的所有边生成语义信息
        results = generator.generate_all_edges(skip_existing=skip_existing)
        print(f"\n生成了 {len(results)} 条语义边")
    elif args.all_missing:
        # 生成所有缺少语义信息的边
        results = generator.generate_all_missing_edges()
        print(f"\n生成了 {len(results)} 条语义边")
    elif args.paper:
        # 根据 paper_id 批量生成相关边
        if args.as_target:
            # 只生成以该论文为 target 的边
            results = generator.generate_edges_for_target(args.paper, skip_existing=skip_existing)
        elif args.as_source:
            # 只生成以该论文为 source 的边
            results = generator.generate_edges_for_source(args.paper, skip_existing=skip_existing)
        else:
            # 生成所有相关的边（source 和 target）
            results = generator.generate_edges_for_paper(args.paper, skip_existing=skip_existing)
        print(f"\n生成了 {len(results)} 条语义边")
    elif args.source and args.target:
        # 生成单条边
        edge_info = generator.generate_edge(args.source, args.target)
        print(f"\n生成的语义边信息:")
        print(json.dumps(edge_info.to_dict(), indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
