"""
TGL 结果收集器

解析 TGL 训练日志，提取指标

日志格式：
    Validation mrr: 0.0388
    Validation mrr: 0.0769
    ...
    Test mrr: 0.1475
    Average test mrr: 0.1475
    Standard deviation: 0.0023

或者：
    Best Validation Metrics:
      MRR: 0.0864
    Test Metrics:
      MRR: 0.0842

功能：
    - 提取所有验证结果（每个 epoch 的验证指标）
    - 计算最佳验证结果（取最大值，适用于 MRR、AP、AUC 等指标）
    - 提取测试结果
    - 提取平均测试结果和标准差
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path


class TGLCollector:
    """TGL 结果收集器"""
    
    # 正则匹配
    # 匹配 "Test METRIC: value"，但不匹配 "Average test METRIC: value"
    TEST_METRIC_PATTERN = re.compile(
        r"(?<!Average\s)Test\s+(\w+)\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配验证结果 "Validation METRIC: value"
    VALIDATION_PATTERN = re.compile(
        r"Validation\s+(\w+)\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配 "Best Validation Metrics:" 后面的指标（多行格式）
    BEST_VALIDATION_HEADER_PATTERN = re.compile(
        r"Best\s+Validation\s+Metrics\s*:",
        re.IGNORECASE
    )
    # 匹配 "Test Metrics:" 后面的指标（多行格式）
    TEST_METRICS_HEADER_PATTERN = re.compile(
        r"Test\s+Metrics\s*:",
        re.IGNORECASE
    )
    # 匹配缩进的指标行（如 "  MRR: 0.0864"）
    INDENTED_METRIC_PATTERN = re.compile(
        r"^\s+(\w+)\s*:\s*([\d.]+)",
        re.IGNORECASE | re.MULTILINE
    )
    AVERAGE_PATTERN = re.compile(
        r"Average\s+test\s+(\w+)\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    STD_PATTERN = re.compile(
        r"Standard\s+deviation\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    MODEL_PATTERN = re.compile(
        r"Running\s+(\w+)\s+on\s+(\S+)",
        re.IGNORECASE
    )
    DATASET_PATTERN = re.compile(
        r"dataset[_\s:]+['\"]?(\S+)['\"]?",
        re.IGNORECASE
    )
    
    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.model: str = ""
        self.dataset: str = ""
        self.test_metrics: Dict[str, float] = {}
        self.validation_metrics: Dict[str, list] = {}  # 存储所有验证结果
        self.best_validation_metrics: Dict[str, float] = {}  # 存储最佳验证结果
        self.average_metrics: Dict[str, float] = {}
        self.std_deviation: Optional[float] = None
        self.status: str = "UNKNOWN"
    
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        # 首先尝试解析多行格式（"Best Validation Metrics:\n  MRR: 0.0864"）
        self._parse_multiline_format()
        
        # 解析验证指标（所有 epoch 的验证结果）
        val_matches = list(self.VALIDATION_PATTERN.finditer(self.log_content))
        for match in val_matches:
            metric_name = match.group(1).upper()
            metric_value = float(match.group(2))
            if metric_name not in self.validation_metrics:
                self.validation_metrics[metric_name] = []
            self.validation_metrics[metric_name].append(metric_value)
        
        # 计算最佳验证结果（取最大值，适用于 MRR、AP、AUC 等指标）
        for metric_name, values in self.validation_metrics.items():
            if values:
                if metric_name not in self.best_validation_metrics:
                    self.best_validation_metrics[metric_name] = max(values)
                else:
                    # 如果已经有多行格式的结果，取两者中的最大值
                    self.best_validation_metrics[metric_name] = max(
                        self.best_validation_metrics[metric_name],
                        max(values)
                    )
        
        # 解析 Test 指标（最后一次运行的结果）
        test_matches = list(self.TEST_METRIC_PATTERN.finditer(self.log_content))
        if test_matches:
            # 取最后一个匹配（通常是最终测试结果）
            for match in test_matches:
                metric_name = match.group(1).upper()
                metric_value = float(match.group(2))
                # 如果已经有多行格式的结果，保留已有的（通常是更准确的）
                if metric_name not in self.test_metrics:
                    self.test_metrics[metric_name] = metric_value
        
        # 解析 Average 指标（多次运行的平均值）
        avg_matches = self.AVERAGE_PATTERN.finditer(self.log_content)
        for match in avg_matches:
            metric_name = match.group(1).upper()
            metric_value = float(match.group(2))
            self.average_metrics[metric_name] = metric_value
        
        # 解析标准差
        std_match = self.STD_PATTERN.search(self.log_content)
        if std_match:
            self.std_deviation = float(std_match.group(1))
        
        # 提取模型和数据集名称
        self._extract_model_dataset()
        
        # 状态：如果有测试指标或平均指标，则认为成功
        self.status = "OK" if (self.test_metrics or self.average_metrics) else "FAILED"
        
        return {
            "model": self.model,
            "dataset": self.dataset,
            "validation_metrics": self.validation_metrics,  # 所有验证结果
            "best_validation_metrics": self.best_validation_metrics,  # 最佳验证结果
            "test_metrics": self.test_metrics,
            "average_metrics": self.average_metrics,
            "std_deviation": self.std_deviation,
            "status": self.status
        }
    
    def _parse_multiline_format(self):
        """解析多行格式的指标（如 "Best Validation Metrics:\n  MRR: 0.0864"）"""
        lines = self.log_content.split('\n')
        in_best_validation = False
        in_test_metrics = False
        
        for i, line in enumerate(lines):
            # 检测 "Best Validation Metrics:" 标题
            if self.BEST_VALIDATION_HEADER_PATTERN.search(line):
                in_best_validation = True
                in_test_metrics = False
                continue
            
            # 检测 "Test Metrics:" 标题
            if self.TEST_METRICS_HEADER_PATTERN.search(line):
                in_test_metrics = True
                in_best_validation = False
                continue
            
            # 如果遇到新的非缩进行，退出当前模式
            if line.strip() and not line.startswith((' ', '\t')):
                if ':' in line and not self.INDENTED_METRIC_PATTERN.match(line):
                    in_best_validation = False
                    in_test_metrics = False
                    continue
            
            # 解析缩进的指标行
            if in_best_validation or in_test_metrics:
                match = self.INDENTED_METRIC_PATTERN.match(line)
                if match:
                    metric_name = match.group(1).upper()
                    metric_value = float(match.group(2))
                    
                    if in_best_validation:
                        # 存储到最佳验证指标
                        self.best_validation_metrics[metric_name] = metric_value
                        # 同时添加到验证指标列表（用于兼容性）
                        if metric_name not in self.validation_metrics:
                            self.validation_metrics[metric_name] = []
                        self.validation_metrics[metric_name].append(metric_value)
                    elif in_test_metrics:
                        # 存储到测试指标
                        self.test_metrics[metric_name] = metric_value
    
    def _extract_model_dataset(self):
        """尝试从日志中提取模型和数据集名称"""
        # 匹配 "Running MODEL on DATASET"
        model_match = self.MODEL_PATTERN.search(self.log_content)
        if model_match:
            self.model = model_match.group(1)
            self.dataset = model_match.group(2)
            return
        
        # 匹配 model: xxx 或 dataset: xxx
        model_match = re.search(r"model[:\s]+['\"]?(\w+)['\"]?", self.log_content, re.IGNORECASE)
        if model_match:
            self.model = model_match.group(1)
        
        dataset_match = self.DATASET_PATTERN.search(self.log_content)
        if dataset_match:
            self.dataset = dataset_match.group(1)
        
        # 尝试从数据集配置中提取
        if not self.dataset:
            dataset_match = re.search(r"dataset_name[:\s=]+['\"]?(\S+)['\"]?", self.log_content, re.IGNORECASE)
            if dataset_match:
                self.dataset = dataset_match.group(1)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取测试集指标（用于 ResultTable）"""
        # 优先返回平均指标（如果有），否则返回单次测试指标
        if self.average_metrics:
            return self.average_metrics
        return self.test_metrics
    
    def get_validation_metrics(self) -> Dict[str, float]:
        """获取最佳验证指标"""
        return self.best_validation_metrics
    
    @classmethod
    def from_file(cls, log_path: Path, model: str = None, dataset: str = None) -> "TGLCollector":
        """从文件读取"""
        content = Path(log_path).read_text(encoding="utf-8")
        collector = cls(content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector
    
    @classmethod
    def from_string(cls, log_content: str, model: str = None, dataset: str = None) -> "TGLCollector":
        """从字符串读取"""
        collector = cls(log_content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector


def parse_tgl_output(log_content: str) -> Dict[str, float]:
    """便捷函数：解析日志并返回指标"""
    collector = TGLCollector.from_string(log_content)
    return collector.get_metrics()


# 测试
if __name__ == "__main__":
    sample_log = """
    Running TGN on tgbl-wiki
    Task type: linkproppred
    Device: cuda
    
    ======================================================================
    Training Summary
    ======================================================================
    Test AP: 0.8234
    Test AUC: 0.9123
    Average test AP: 0.8156
    Standard deviation: 0.0023
    ======================================================================
    """
    
    collector = TGLCollector.from_file("/public/home/maoyaoxin/zyg/GraphLearning/TGL/results/saved_log/TGN_tgbl-wiki_20260103_221433.log")
    result = collector.collect()
    print(f"Model: {collector.model}")
    print(f"Dataset: {collector.dataset}")
    print(f"Status: {result['status']}")
    print(f"Validation Metrics (all): {result['validation_metrics']}")
    print(f"Best Validation Metrics: {result['best_validation_metrics']}")
    print(f"Test Metrics: {result['test_metrics']}")
    print(f"Average Metrics: {result['average_metrics']}")
    print(f"Std Deviation: {result['std_deviation']}")

