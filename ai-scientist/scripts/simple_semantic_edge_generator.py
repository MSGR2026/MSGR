"""
简化的语义边生成器

使用方法:
python simple_semantic_edge_generator.py --source Informer_2020 --target Autoformer_2021 \
    --domain TimeSeries --task long_term_forecast
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# 添加 src 目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent  # ai-scientist/scripts/
AI_SCIENTIST_DIR = SCRIPT_DIR.parent  # ai-scientist/
SRC_DIR = AI_SCIENTIST_DIR / "src"  # ai-scientist/src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class SemanticEdgeResult:
    """语义边生成结果"""
    source: str
    target: str
    similarities: str
    differences: str


class SimpleSemanticEdgeGenerator:
    """简化的语义边生成器
    
    只生成单条边，基于：
    - 源论文的方法+示例代码
    - 目标论文的方法
    
    输出similarities和differences到JSON文件
    """
    
    # 提示词模板路径 (相对于 src/ 目录)
    TEMPLATE_FILE = SRC_DIR / "agent" / "prompt" / "semantic_edge_template.jinja"
    
    def __init__(
        self,
        llm_client,
        local_graph_loader,
        domain: str,
        task: str,
        output_dir: Optional[Path] = None,
    ):
        """初始化生成器
        
        Args:
            llm_client: LLM客户端实例
            local_graph_loader: LocalGraphLoader实例
            domain: 领域名称
            task: 任务名称
            output_dir: 输出目录，默认为 output/semantic_edges
        """
        self.llm = llm_client
        self.loader = local_graph_loader
        self.domain = domain
        self.task = task
        
        if output_dir is None:
            output_dir = AI_SCIENTIST_DIR / "output" / "semantic_edges"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Jinja2 环境
        self._init_jinja_env()
    
    def _init_jinja_env(self) -> None:
        """初始化 Jinja2 模板环境"""
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
        except ImportError:
            raise ImportError("请安装 jinja2: pip install jinja2")
        
        template_dir = self.TEMPLATE_FILE.parent
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(disabled_extensions=('jinja',)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    
    def _load_template(self):
        """加载 Jinja2 模板"""
        return self.jinja_env.get_template(self.TEMPLATE_FILE.name)
    
    def _collect_context(
        self,
        source_paper_id: str,
        target_paper_id: str,
    ) -> Dict[str, Any]:
        """收集生成语义边所需的上下文数据
        
        只收集：
        - 源论文：方法 + 示例代码
        - 目标论文：方法
        """
        # 获取源节点信息
        source_method = self.loader.get_method(source_paper_id)
        source_algorithm = self.loader.get_algorithm_implementation(source_paper_id)
        source_info = self.loader.get_paper_info(source_paper_id)
        
        # 获取目标节点信息（只需要方法部分）
        target_method = self.loader.get_method(target_paper_id)
        target_info = self.loader.get_paper_info(target_paper_id)
        
        return {
            "domain": self.domain,
            "task": self.task,
            # 源节点信息
            "source_paper_id": source_paper_id,
            "source_paper_title": source_info.get("paper_title", "") if source_info else "",
            "source_year": source_info.get("year", 0) if source_info else 0,
            "source_method": source_method,
            "source_algorithm_code": source_algorithm,
            # 目标节点信息
            "target_paper_id": target_paper_id,
            "target_paper_title": target_info.get("paper_title", "") if target_info else "",
            "target_year": target_info.get("year", 0) if target_info else 0,
            "target_method": target_method,
        }
    
    def _render_prompt(self, context: Dict[str, Any]) -> tuple[str, str]:
        """渲染提示词模板
        
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
    
    def _parse_llm_response(
        self,
        response: str,
        source_id: str,
        target_id: str
    ) -> SemanticEdgeResult:
        """解析 LLM 响应
        
        期望的JSON格式:
        {
            "similarities": "相同点描述",
            "differences": "不同点描述"
        }
        """
        import re
        
        # 查找 JSON 代码块
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        json_match = re.search(json_pattern, response)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return SemanticEdgeResult(
                    source=source_id,
                    target=target_id,
                    similarities=data.get("similarities", ""),
                    differences=data.get("differences", ""),
                )
            except json.JSONDecodeError as e:
                print(f"[警告] JSON解析失败: {e}")
        
        # 如果没有JSON，尝试从文本中解析
        similarities = ""
        differences = ""
        
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
        
        return SemanticEdgeResult(
            source=source_id,
            target=target_id,
            similarities=similarities,
            differences=differences,
        )
    
    def generate_edge(
        self,
        source_paper_id: str,
        target_paper_id: str,
    ) -> SemanticEdgeResult:
        """生成两个论文之间的语义边信息
        
        Args:
            source_paper_id: 边起点论文ID（先前工作）
            target_paper_id: 边终点论文ID（后续工作）
        
        Returns:
            SemanticEdgeResult 对象
        """
        print(f"\n{'='*60}")
        print(f"生成语义边: {source_paper_id} -> {target_paper_id}")
        print(f"{'='*60}\n")
        
        # 收集上下文
        print("[1/4] 收集论文数据...")
        context = self._collect_context(source_paper_id, target_paper_id)
        
        # 渲染提示词
        print("[2/4] 渲染提示词...")
        system_prompt, user_prompt = self._render_prompt(context)
        print(f"  - System prompt: {len(system_prompt)} 字符")
        print(f"  - User prompt: {len(user_prompt)} 字符")
        
        # 打印渲染后的提示词
        print("\n" + "="*60)
        print("System Prompt:")
        print("="*60)
        print(system_prompt[:1000])
        print("\n" + "="*60)
        print("User Prompt:")
        print("="*60)
        print(user_prompt[:8000])
        print("="*60 + "\n")
        
        # 调用 LLM
        print("[3/4] 调用大模型...")
        response = self.llm.generate(system_prompt, user_prompt)
        
        # 打印 Token 统计
        print("\n" + "="*60)
        print("Token 使用情况:")
        print("="*60)
        self.llm.print_token_stats()
        print("="*60 + "\n")
        
        # 解析响应
        print("[4/4] 解析响应...")
        edge_result = self._parse_llm_response(response, source_paper_id, target_paper_id)
        
        # 保存结果
        self._save_results(edge_result, response)
        
        print(f"\n✓ 语义边生成完成!")
        print(f"  - 相同点: {len(edge_result.similarities)} 字符")
        print(f"  - 不同点: {len(edge_result.differences)} 字符")
        
        return edge_result
    
    def _save_results(self, result: SemanticEdgeResult, raw_response: str) -> None:
        """保存生成结果到文件
        
        生成以下文件:
        1. {source}_{target}_edge.json - 边信息
        2. {source}_{target}_raw_response.txt - 原始响应
        """
        base_name = f"{result.source}_{result.target}"
        
        # 1. 保存边信息（JSON）
        edge_data = {
            "source": result.source,
            "target": result.target,
            "type": "in-domain",
            "similarities": result.similarities,
            "differences": result.differences,
        }
        edge_file = self.output_dir / f"{base_name}_edge.json"
        with open(edge_file, "w", encoding="utf-8") as f:
            json.dump(edge_data, f, ensure_ascii=False, indent=2)
        print(f"\n保存边信息到: {edge_file}")
        
        # 2. 保存原始响应（TXT）
        raw_file = self.output_dir / f"{base_name}_raw_response.txt"
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(raw_response)
        print(f"保存原始响应到: {raw_file}")


def main():
    """命令行入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="简化的语义边生成器 - 生成单条边",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python simple_semantic_edge_generator.py \\
      --source Informer_2020 --target Autoformer_2021 \\
      --domain TimeSeries --task long_term_forecast
        """
    )
    parser.add_argument("--source", type=str, required=True, help="源论文ID(先前工作)")
    parser.add_argument("--target", type=str, required=True, help="目标论文ID(后续工作)")
    parser.add_argument("--domain", type=str, required=True, help="领域名称 (如: TimeSeries, Recsys)")
    parser.add_argument("--task", type=str, required=True, help="任务名称 (如: long_term_forecast)")
    parser.add_argument("--config", type=str, default=str(AI_SCIENTIST_DIR / "configs" / "llm.yaml"), help="LLM 配置文件路径")
    parser.add_argument("--output-dir", type=str, help="输出目录路径")
    
    args = parser.parse_args()
    
    # 导入必要模块
    from llm.client import LLMClient
    from graph.local_graph.local_graph_loader import LocalGraphLoader
    
    # 初始化
    print("初始化LLM客户端...")
    config_path = Path(args.config) if not Path(args.config).is_absolute() else args.config
    llm_client = LLMClient(config_path=str(config_path))
    
    print(f"加载图数据 (领域: {args.domain}, 任务: {args.task})...")
    # 显式指定 data_root 为 ai-scientist/data
    data_root = AI_SCIENTIST_DIR / "data"
    loader = LocalGraphLoader(domain=args.domain, task=args.task, data_root=data_root)
    print(f"  - 数据根目录: {data_root}")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    generator = SimpleSemanticEdgeGenerator(
        llm_client=llm_client,
        local_graph_loader=loader,
        domain=args.domain,
        task=args.task,
        output_dir=output_dir,
    )
    
    # 生成边
    result = generator.generate_edge(args.source, args.target)
    
    print("\n" + "="*60)
    print("生成完成!")
    print("="*60)


if __name__ == "__main__":
    main()
