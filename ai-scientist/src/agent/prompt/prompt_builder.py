"""
PromptBuilder - 提示词构建器

本模块负责加载 Jinja2 模板并构建完整的 LLM 提示词。

功能特性:
    1. 模板加载: 根据 domain/task 加载对应的 Jinja2 模板
    2. 上下文收集: 从 LocalGraph 收集论文内容、邻居信息
    3. 全局知识注入: 整合 GlobalGraph 中的领域知识
    4. 历史记录: 将之前的尝试结果作为上下文
    5. 提示词渲染: 使用 Jinja2 渲染最终提示词

模板类型:
    - algorithm.jinja: 算法代码生成模板
    - hyperparameter.jinja: 超参数配置生成模板

作者: GraphScientist Team
版本: 2.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Jinja2 模板引擎
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
except ImportError:
    raise ImportError("请安装 jinja2: pip install jinja2")

# 添加 src 目录到路径
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.session import RoundResult

if TYPE_CHECKING:
    from graph.local_graph.local_graph_loader import LocalGraphLoader


class TemplateNotFoundError(Exception):
    """
    模板文件未找到异常
    
    当指定的模板文件不存在时抛出此异常。
    """
    pass


class PromptBuilder:
    """
    提示词构建器
    
    负责加载模板、收集上下文、渲染提示词。
    
    属性:
        domain (str): 领域名称，如 "Recsys"
        task (str): 任务名称，如 "MultiModal"
        loader (LocalGraphLoader): 本地图加载器，用于获取论文内容
        global_graph: 全局图，包含领域知识 (可选)
        template_dir (Path): 模板目录路径
        env (Environment): Jinja2 模板环境
    
    支持的模板类型:
        - algorithm: 算法代码生成
        - hyperparameter: 超参数配置生成
    
    使用示例:
        >>> builder = PromptBuilder(
        ...     domain="Recsys",
        ...     task="MultiModal",
        ...     local_graph_loader=loader
        ... )
        >>> system_prompt, user_prompt = builder.build_algorithm_prompt("BM3_2023")
    """
    
    # 默认模板目录 (相对于本文件)
    DEFAULT_TEMPLATE_DIR = Path(__file__).parent
    
    # 任务名称映射: 将标准任务名映射到目录名
    TASK_MAPPING = {
        "MultiModalRecommendation": "MultiModal",
        "MultiModal": "MultiModal",
        # 可以添加更多映射
    }
    
    # 模板分隔符标记
    SYSTEM_PROMPT_MARKER = "---SYSTEM_PROMPT---"
    USER_PROMPT_MARKER = "---USER_PROMPT---"
    
    def __init__(
        self,
        domain: str,
        task: str,
        local_graph_loader: "LocalGraphLoader",
        global_graph: Optional[Any] = None,
        template_dir: Optional[Path] = None,
    ):
        """
        初始化提示词构建器
        
        参数:
            domain: 领域名称 (如 "Recsys")
            task: 任务名称 (如 "MultiModal" 或 "MultiModalRecommendation")
            local_graph_loader: LocalGraphLoader 实例，用于获取论文数据
            global_graph: 可选的 GlobalGraph 实例，包含领域知识
            template_dir: 可选的模板目录路径，默认使用本模块目录
        
        异常:
            FileNotFoundError: 如果指定的模板目录不存在
        """
        self.domain = domain
        # 将任务名映射到目录名
        self.task = self.TASK_MAPPING.get(task, task)
        self.loader = local_graph_loader
        self.global_graph = global_graph
        
        # 设置模板目录
        self.template_dir = template_dir or self.DEFAULT_TEMPLATE_DIR
        if not self.template_dir.exists():
            raise FileNotFoundError(f"模板目录不存在: {self.template_dir}")
        
        # 验证必要的模板文件存在
        self._validate_templates()
        
        # 初始化 Jinja2 环境
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(disabled_extensions=('jinja',)),
            trim_blocks=True,       # 去除块标签后的第一个换行符
            lstrip_blocks=True,     # 去除块标签前的空白
        )
        
        # 添加自定义过滤器
        self._register_custom_filters()
    
    def _validate_templates(self) -> None:
        """
        验证必要的模板文件是否存在
        
        检查以下模板是否存在:
            - algorithm.jinja: 算法代码生成
            - hyperparameter.jinja: 超参数配置生成
        
        异常:
            TemplateNotFoundError: 如果任何必要模板缺失
        """
        required_templates = [
            "algorithm",
            "hyperparameter",
        ]
        missing = []
        
        for template_type in required_templates:
            template_path = self.template_dir / self.domain / self.task / f"{template_type}.jinja"
            if not template_path.exists():
                missing.append(str(template_path))
        
        if missing:
            raise TemplateNotFoundError(
                f"以下模板文件缺失，请创建:\n" + "\n".join(f"  - {p}" for p in missing)
            )
    
    def _register_custom_filters(self) -> None:
        """
        注册 Jinja2 自定义过滤器
        
        过滤器列表:
            - truncate_text: 截断文本到指定长度
        """
        def truncate_text(text: str, length: int = 1000, suffix: str = "...") -> str:
            """截断文本到指定长度"""
            if not text:
                return ""
            if len(text) <= length:
                return text
            return text[:length] + suffix
        
        self.env.filters['truncate_text'] = truncate_text
    
    def _get_template_path(self, template_type: str) -> str:
        """
        获取模板文件路径
        
        参数:
            template_type: 模板类型 ("algorithm", "hyperparameter")
        
        返回:
            模板文件的相对路径 (相对于 template_dir)
        
        示例:
            >>> builder._get_template_path("algorithm")
            "Recsys/MultiModal/algorithm.jinja"
        """
        return f"{self.domain}/{self.task}/{template_type}.jinja"
    
    def _render_template(
        self,
        template_type: str,
        context: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        渲染模板并返回 system_prompt 和 user_prompt
        
        参数:
            template_type: 模板类型
            context: 上下文数据
        
        返回:
            (system_prompt, user_prompt) 元组
        
        异常:
            TemplateNotFoundError: 模板文件不存在
            jinja2.TemplateError: 模板渲染错误
        
        模板格式:
            模板使用标记分隔符来区分不同的提示词部分:
            ---SYSTEM_PROMPT---
            系统提示词内容...
            ---USER_PROMPT---
            用户提示词内容...
        """
        template_path = self._get_template_path(template_type)
        
        try:
            template = self.env.get_template(template_path)
        except TemplateNotFound:
            raise TemplateNotFoundError(
                f"模板文件不存在: {self.template_dir / template_path}\n"
                f"请在 {self.template_dir}/{self.domain}/{self.task}/ 目录下创建 {template_type}.jinja 模板文件"
            )
        
        # 渲染整个模板
        rendered = template.render(**context)
        
        # 使用标记分隔符解析不同的提示词部分
        return self._extract_prompt_sections(rendered)
    
    def _extract_prompt_sections(self, rendered: str) -> Tuple[str, str]:
        """
        从渲染后的模板中提取 system_prompt 和 user_prompt
        
        参数:
            rendered: 完整的渲染后模板
        
        返回:
            (system_prompt, user_prompt) 元组
        
        异常:
            ValueError: 如果模板格式不正确 (缺少分隔符)
        """
        # 检查是否包含分隔符
        has_system = self.SYSTEM_PROMPT_MARKER in rendered
        has_user = self.USER_PROMPT_MARKER in rendered
        
        if not has_system or not has_user:
            raise ValueError(
                f"模板格式错误: 必须同时包含 {self.SYSTEM_PROMPT_MARKER} 和 {self.USER_PROMPT_MARKER} 分隔符"
            )
        
        # 解析分隔符
        parts = rendered.split(self.USER_PROMPT_MARKER)
        user_prompt = parts[1].strip() if len(parts) > 1 else ""
        
        system_part = parts[0]
        system_prompt = system_part.replace(self.SYSTEM_PROMPT_MARKER, "").strip()
        
        return system_prompt, user_prompt
    
    def _collect_algorithm_context(
        self,
        paper_id: str,
        history: Optional[List[RoundResult]] = None,
        current_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        收集算法生成模板所需的上下文数据
        
        参数:
            paper_id: 目标论文ID
            history: 历史尝试记录
            current_code: 当前算法代码（用于迭代改进）
        
        返回:
            包含所有模板变量的字典
        """
        context = {
            "paper_id": paper_id,
            "domain": self.domain,
            "task": self.task,
            "method": self.loader.get_method(paper_id),
            "idea": self.loader.get_idea(paper_id),
            "introduction": self.loader.get_introduction(paper_id),
            "neighbors": self.loader.get_neighbors(paper_id, k=1, edge_type="in-domain"),
            "history": history or [],
            "current_code": current_code,
        }
        
        # 添加全局领域知识
        if self.global_graph and hasattr(self.global_graph, 'to_context_string'):
            context["global_context"] = self.global_graph.to_context_string()
        else:
            context["global_context"] = None
        
        return context
    
    def _collect_hyperparameter_context(
        self,
        paper_id: str,
        algorithm_code: str,
        history: Optional[List[RoundResult]] = None,
        current_hyperparameter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        收集超参数生成模板所需的上下文数据
        
        参数:
            paper_id: 目标论文ID
            algorithm_code: 已生成的算法代码
            history: 历史尝试记录
            current_hyperparameter: 当前超参数配置（用于迭代改进）
        
        返回:
            包含所有模板变量的字典
        """
        context = {
            "paper_id": paper_id,
            "domain": self.domain,
            "task": self.task,
            "algorithm_code": algorithm_code,
            "hyperparameter_desc": self.loader.get_hyperparameter(paper_id),
            "neighbors": self.loader.get_neighbors(paper_id, k=1, edge_type="in-domain"),
            "history": history or [],
            "current_hyperparameter": current_hyperparameter,
        }
        
        # 添加全局领域知识
        if self.global_graph and hasattr(self.global_graph, 'to_context_string'):
            context["global_context"] = self.global_graph.to_context_string()
        else:
            context["global_context"] = None
        
        return context
    
    def build_algorithm_prompt(
        self,
        paper_id: str,
        history: Optional[List[RoundResult]] = None,
        current_code: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        构建算法生成的完整提示词
        
        使用 algorithm.jinja 模板渲染提示词。
        
        参数:
            paper_id: 目标论文ID
            history: 可选的历史尝试记录
            current_code: 当前算法代码（用于迭代改进）
        
        返回:
            (system_prompt, user_prompt) 元组
        
        异常:
            TemplateNotFoundError: 模板文件不存在
        
        示例:
            >>> system_prompt, user_prompt = builder.build_algorithm_prompt("BM3_2023")
            >>> response = llm.generate(system_prompt, user_prompt)
        """
        # 收集上下文
        context = self._collect_algorithm_context(paper_id, history, current_code)
        
        # 渲染模板
        system_prompt, user_prompt = self._render_template("algorithm", context)
        
        # 打印调试信息
        print(f"[PromptBuilder] Algorithm prompt: system={len(system_prompt)} chars, user={len(user_prompt)} chars")
        
        return system_prompt, user_prompt
    
    def build_hyperparameter_prompt(
        self,
        paper_id: str,
        algorithm_code: str,
        history: Optional[List[RoundResult]] = None,
        current_hyperparameter: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        构建超参数生成的完整提示词
        
        使用 hyperparameter.jinja 模板渲染提示词。
        
        参数:
            paper_id: 目标论文ID
            algorithm_code: 已生成的算法代码
            history: 可选的历史尝试记录
            current_hyperparameter: 当前超参数配置（用于迭代改进）
        
        返回:
            (system_prompt, user_prompt) 元组
        
        异常:
            TemplateNotFoundError: 模板文件不存在
        
        示例:
            >>> system_prompt, user_prompt = builder.build_hyperparameter_prompt(
            ...     paper_id="BM3_2023",
            ...     algorithm_code=generated_code
            ... )
            >>> response = llm.generate(system_prompt, user_prompt)
        """
        # 收集上下文
        context = self._collect_hyperparameter_context(paper_id, algorithm_code, history, current_hyperparameter)
        
        # 渲染模板
        system_prompt, user_prompt = self._render_template("hyperparameter", context)
        
        # 打印调试信息
        print(f"[PromptBuilder] Hyperparameter prompt: system={len(system_prompt)} chars, user={len(user_prompt)} chars")
        
        return system_prompt, user_prompt
