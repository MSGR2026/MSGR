"""
配置管理模块

提供统一的配置加载、验证和访问功能。

特性:
- 支持 YAML 配置文件
- 支持环境变量替换 (${ENV_VAR})
- 支持配置文件引用 (!include)
- 支持命令行参数覆盖 (--config.key=value)
- 自动参数校验
- 提供合理的默认值

使用示例:
    from common.config import Config, load_config
    
    # 加载配置
    config = load_config("ai-scientist/configs/config.yaml")
    
    # 访问配置
    config.task.domain       # "Recsys"
    config.agent.max_rounds  # 10
    config.paths.data_root   # "data"
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml


# ==================== 工具函数 ====================

def get_repo_root() -> Path:
    """Get the repository root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "graph_out.json").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "ai-scientist").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent.parent


def get_config_dir() -> Path:
    """获取配置文件目录"""
    return get_repo_root() / "ai-scientist" / "configs"


# 环境变量匹配模式
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """递归替换配置中的环境变量 (${VAR_NAME} 格式)"""
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            env_name = match.group(1)
            return os.getenv(env_name, "")
        return _ENV_VAR_PATTERN.sub(_replace, value)
    return value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度合并两个字典，override 覆盖 base"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: Dict, keys: List[str], value: Any) -> None:
    """设置嵌套字典的值"""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    解析命令行参数覆盖
    
    格式: --config.key.subkey=value 或 --key.subkey=value
    """
    overrides = {}
    for arg in args:
        if arg.startswith("--") and "=" in arg:
            key_part, value = arg[2:].split("=", 1)
            # 移除 config. 前缀（如果有）
            if key_part.startswith("config."):
                key_part = key_part[7:]
            keys = key_part.split(".")
            # 尝试解析值类型
            parsed_value = _parse_value(value)
            _set_nested(overrides, keys, parsed_value)
    return overrides


def _parse_value(value: str) -> Any:
    """解析字符串值为适当的类型"""
    # 布尔值
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    # 数字
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    # 列表 (逗号分隔)
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]
    return value


# ==================== 配置数据类 ====================

@dataclass
class TaskConfig:
    """任务配置"""
    domain: str = "Recsys"
    task_name: str = "MultiModalRecommendation"
    paper_id: Union[str, List[str]] = ""
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskConfig":
        paper_id = d.get("paper_id", "")
        if paper_id is None:
            paper_id = ""
        return cls(
            domain=d.get("domain", "Recsys"),
            task_name=d.get("task_name", "MultiModalRecommendation"),
            paper_id=paper_id,
        )

    def paper_ids(self) -> List[str]:
        """Normalize paper_id to a list."""
        if isinstance(self.paper_id, list):
            return [p for p in self.paper_id if p]
        if isinstance(self.paper_id, str) and self.paper_id.strip():
            return [self.paper_id.strip()]
        return []


@dataclass
class AgentConfig:
    """Agent 配置"""
    max_rounds: int = 10
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentConfig":
        return cls(
            max_rounds=int(d.get("max_rounds", 10)),
        )


@dataclass
class ResourceConfig:
    """资源配置"""
    gpu_ids: Optional[List[int]] = None
    memory_limit: str = "16g"
    cpu_limit: Optional[int] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResourceConfig":
        gpu_ids = d.get("gpu_ids")
        if isinstance(gpu_ids, str):
            gpu_ids = [int(g.strip()) for g in gpu_ids.split(",") if g.strip()]
        elif isinstance(gpu_ids, list):
            gpu_ids = [int(g) for g in gpu_ids]
        return cls(
            gpu_ids=gpu_ids,
            memory_limit=d.get("memory_limit", "16g"),
            cpu_limit=d.get("cpu_limit"),
        )


@dataclass
class PathConfig:
    """路径配置"""
    data_root: str = "data"
    output_dir: str = "ai-scientist/output"
    log_dir: str = "ai-scientist/logs"
    paperbench_root: str = "paperbench_pro"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PathConfig":
        return cls(
            data_root=d.get("data_root", "data"),
            output_dir=d.get("output_dir", "ai-scientist/output"),
            log_dir=d.get("log_dir", "ai-scientist/logs"),
            paperbench_root=d.get("paperbench_root", "paperbench_pro"),
        )
    
    def resolve_data_root(self) -> Path:
        """解析数据根目录为绝对路径"""
        env_root = os.environ.get("GRAPH_SCIENTIST_DATA_ROOT")
        if env_root:
            return Path(env_root).resolve()
        path = Path(self.data_root)
        if path.is_absolute():
            return path
        return get_repo_root() / path
    
    def resolve_output_dir(self) -> Path:
        """解析输出目录为绝对路径"""
        path = Path(self.output_dir)
        if path.is_absolute():
            return path
        return get_repo_root() / path
    
    def resolve_log_dir(self) -> Path:
        """解析日志目录为绝对路径"""
        path = Path(self.log_dir)
        if path.is_absolute():
            return path
        return get_repo_root() / path
    
    def resolve_paperbench_root(self) -> Path:
        """解析 PaperBench-Pro 根目录为绝对路径"""
        path = Path(self.paperbench_root)
        if path.is_absolute():
            return path
        return get_repo_root() / path


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "claude"
    model_name: str = "claude-sonnet-4-20250514"
    base_url: str = "https://api.anthropic.com/v1"
    api_key: str = ""
    api_version: str = "2023-06-01"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    retry_count: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    max_concurrency: int = 5
    pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=d.get("provider", "claude"),
            model_name=d.get("model_name", "claude-sonnet-4-20250514"),
            base_url=d.get("base_url", "https://api.anthropic.com/v1"),
            api_key=d.get("api_key", ""),
            api_version=d.get("api_version", "2023-06-01"),
            max_tokens=int(d.get("max_tokens", 4096)),
            temperature=float(d.get("temperature", 0.7)),
            timeout=int(d.get("timeout", 60)),
            retry_count=int(d.get("retry_count", 3)),
            retry_base_delay=float(d.get("retry_base_delay", 1.0)),
            retry_max_delay=float(d.get("retry_max_delay", 60.0)),
            max_concurrency=int(d.get("max_concurrency", 5)),
            pricing=d.get("pricing", {}),
        )


@dataclass
class DockerImageConfig:
    """Docker 镜像配置"""
    name: str = ""
    dockerfile: str = ""
    status: str = "pending"
    conda_env: str = ""
    python_cmd: str = "python"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DockerImageConfig":
        return cls(
            name=d.get("name", ""),
            dockerfile=d.get("dockerfile", ""),
            status=d.get("status", "pending"),
            conda_env=d.get("conda_env", ""),
            python_cmd=d.get("python_cmd", "python"),
        )


@dataclass
class DockerConfig:
    """Docker 配置"""
    enabled: bool = True
    images: Dict[str, str] = field(default_factory=dict)
    default_image: str = "ubuntu:22.04"
    memory_limit: str = "16g"
    cpu_limit: Optional[int] = None
    timeout_command: int = 300
    timeout_training: int = 3600
    timeout_startup: int = 120
    workspace: str = "/workspace"
    tools_dir: str = "/agent_tools"
    logs_dir: str = "/workspace/logs"
    interactive: bool = True
    shell: str = "/bin/bash"
    docker_tools: List[str] = field(default_factory=list)
    default_timeout: int = 120
    max_output_length: int = 50000
    built_images: Dict[str, DockerImageConfig] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DockerConfig":
        docker = d.get("docker", d)
        resources = docker.get("resources", {})
        timeout = docker.get("timeout", {})
        paths = docker.get("paths", {})
        session = docker.get("session", {})
        tools = d.get("tools", {})
        
        built_images = {}
        for name, img_data in d.get("built_images", {}).items():
            if isinstance(img_data, dict):
                built_images[name] = DockerImageConfig.from_dict(img_data)
        
        return cls(
            enabled=docker.get("enabled", True),
            images=docker.get("images", {}),
            default_image=docker.get("default_image", "ubuntu:22.04"),
            memory_limit=resources.get("memory_limit", "16g"),
            cpu_limit=resources.get("cpu_limit"),
            timeout_command=timeout.get("command", 300),
            timeout_training=timeout.get("training", 3600),
            timeout_startup=timeout.get("startup", 120),
            workspace=paths.get("workspace", "/workspace"),
            tools_dir=paths.get("tools", "/agent_tools"),
            logs_dir=paths.get("logs", "/workspace/logs"),
            interactive=session.get("interactive", True),
            shell=session.get("shell", "/bin/bash"),
            docker_tools=tools.get("docker_tools", ["bash", "str_replace_based_edit_tool", "json_edit_tool"]),
            default_timeout=tools.get("default_timeout", 120),
            max_output_length=tools.get("max_output_length", 50000),
            built_images=built_images,
        )
    
    def get_image_for_domain(self, domain: str) -> str:
        """获取指定领域的 Docker 镜像"""
        return self.images.get(domain, self.default_image)


@dataclass 
class DomainConfig:
    """领域配置"""
    subdir: str = ""
    model_dir: str = ""
    config_dir: str = ""
    script: str = "replicate.py"
    default_dataset: str = ""
    default_task: str = ""
    metric_key: str = "accuracy"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainConfig":
        return cls(
            subdir=d.get("subdir", ""),
            model_dir=d.get("model_dir", ""),
            config_dir=d.get("config_dir", ""),
            script=d.get("script", "replicate.py"),
            default_dataset=d.get("default_dataset", ""),
            default_task=d.get("default_task", ""),
            metric_key=d.get("metric_key", "accuracy"),
        )


@dataclass
class PaperBenchProConfig:
    """PaperBench-Pro 配置"""
    root: str = "paperbench_pro"
    timeout: int = 3600
    domains: Dict[str, DomainConfig] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperBenchProConfig":
        domains = {}
        for name, domain_data in d.get("domains", {}).items():
            if isinstance(domain_data, dict):
                domains[name] = DomainConfig.from_dict(domain_data)
        
        return cls(
            root=d.get("root", "paperbench_pro"),
            timeout=int(d.get("timeout", 3600)),
            domains=domains,
        )
    
    def get_domain_config(self, domain: str) -> Optional[DomainConfig]:
        """获取指定领域的配置"""
        return self.domains.get(domain)


# ==================== 主配置类 ====================

@dataclass
class Config:
    """
    统一配置类
    
    包含所有子配置模块的引用。
    """
    task: TaskConfig = field(default_factory=TaskConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    paperbench_pro: PaperBenchProConfig = field(default_factory=PaperBenchProConfig)
    
    # 原始配置数据
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        return cls(
            task=TaskConfig.from_dict(d.get("task", {})),
            agent=AgentConfig.from_dict(d.get("agent", {})),
            resources=ResourceConfig.from_dict(d.get("resources", {})),
            paths=PathConfig.from_dict(d.get("paths", {})),
            llm=LLMConfig.from_dict(d.get("llm", {})),
            docker=DockerConfig.from_dict(d.get("docker", d)),  # 兼容旧格式
            paperbench_pro=PaperBenchProConfig.from_dict(d.get("paperbench_pro", {})),
            _raw=d,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._raw.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点分隔的键）"""
        keys = key.split(".")
        value = self._raw
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value


# ==================== 配置加载器 ====================

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or get_config_dir()
        self._include_pattern = re.compile(r"^!include\s+(.+)$")
    
    def load(self, config_path: Optional[Union[str, Path]] = None) -> Config:
        """
        加载配置
        
        Args:
            config_path: 配置文件路径。如果为 None，则使用默认配置。
            
        Returns:
            Config 实例
        """
        # 确定配置文件路径
        if config_path is None:
            config_path = self.config_dir / "config.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = get_repo_root() / config_path
        
        # 加载配置
        if config_path.exists():
            data = self._load_yaml_with_includes(config_path)
        else:
            # 如果主配置不存在，尝试加载各个独立配置
            data = self._load_separate_configs()
        
        # 环境变量替换
        data = _expand_env_vars(data)
        
        return Config.from_dict(data)
    
    def load_with_overrides(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cli_args: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Config:
        """
        加载配置并应用覆盖
        
        Args:
            config_path: 配置文件路径
            cli_args: 命令行参数列表（用于解析 --key=value 格式）
            overrides: 直接覆盖的字典
            
        Returns:
            Config 实例
        """
        # 加载基础配置
        config = self.load(config_path)
        data = config._raw.copy()
        
        # 应用命令行覆盖
        if cli_args:
            cli_overrides = _parse_cli_overrides(cli_args)
            data = _deep_merge(data, cli_overrides)
        
        # 应用直接覆盖
        if overrides:
            data = _deep_merge(data, overrides)
        
        return Config.from_dict(data)
    
    def _load_yaml_with_includes(self, path: Path) -> Dict[str, Any]:
        """加载 YAML 文件，支持 !include 指令"""
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 处理 !include 指令
        lines = content.split("\n")
        processed_lines = []
        for line in lines:
            match = self._include_pattern.match(line.strip())
            if match:
                include_file = match.group(1).strip()
                include_path = path.parent / include_file
                if include_path.exists():
                    included_data = self._load_yaml_with_includes(include_path)
                    # 将包含的内容转换为 YAML 字符串
                    if included_data:
                        processed_lines.append(yaml.dump(included_data, default_flow_style=False))
                continue
            processed_lines.append(line)
        
        processed_content = "\n".join(processed_lines)
        data = yaml.safe_load(processed_content)
        return data if isinstance(data, dict) else {}
    
    def _load_separate_configs(self) -> Dict[str, Any]:
        """加载各个独立的配置文件"""
        data = {}
        
        # LLM 配置
        llm_path = self.config_dir / "llm.yaml"
        if llm_path.exists():
            llm_data = self._load_yaml_with_includes(llm_path)
            if "llm" in llm_data:
                data["llm"] = llm_data["llm"]
            else:
                data["llm"] = llm_data
        
        # Storage 配置
        storage_path = self.config_dir / "storage.yaml"
        if storage_path.exists():
            storage_data = self._load_yaml_with_includes(storage_path)
            if "data_root" in storage_data:
                data.setdefault("paths", {})["data_root"] = storage_data["data_root"]
        
        # Agent 配置
        agent_path = self.config_dir / "agent.yaml"
        if agent_path.exists():
            agent_data = self._load_yaml_with_includes(agent_path)
            if "agent" in agent_data:
                data["agent"] = agent_data["agent"]
            elif agent_data:
                data["agent"] = agent_data
        
        # PaperBench-Pro 配置由其自身管理 (paperbench_pro/configs/)
        
        return data


# ==================== 便捷函数 ====================

_default_config: Optional[Config] = None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    cli_args: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    加载配置的便捷函数
    
    Args:
        config_path: 配置文件路径。None 则使用默认路径。
        cli_args: 命令行参数覆盖
        overrides: 字典覆盖
        
    Returns:
        Config 实例
    """
    loader = ConfigLoader()
    return loader.load_with_overrides(config_path, cli_args, overrides)


def get_config() -> Config:
    """获取全局配置实例（懒加载）"""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config


def set_config(config: Config) -> None:
    """设置全局配置实例"""
    global _default_config
    _default_config = config


def reset_config() -> None:
    """重置全局配置"""
    global _default_config
    _default_config = None


# ==================== 兼容旧 API ====================

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def resolve_data_root(data_root: Optional[str] = None) -> Path:
    """
    Resolve the data root path (兼容旧 API).
    
    Args:
        data_root: Optional data root from config. Can be relative or absolute.
        
    Returns:
        Resolved absolute path to data root.
    """
    env_root = os.environ.get("GRAPH_SCIENTIST_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    
    if data_root is None:
        data_root = "data"
    
    path = Path(data_root)
    if path.is_absolute():
        return path
    
    return get_repo_root() / path
