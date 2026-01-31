"""
GSL 结果收集器

解析 GSL Benchmark 训练日志，提取指标

日志格式：
    每个实验的结果：
        Loss(test) 1.3256 | Acc(test) 0.8160
        Loss(test) 1.9303 | Acc(test) 0.8130
        ...
    
    最终统计：
        All runs:
        Highest Train: 88.71 ± 7.32
        Highest Valid: 81.84 ± 0.87
"""
import re
from typing import Dict, Any, Optional, List
from pathlib import Path


class GSLCollector:
    """GSL Benchmark 结果收集器"""
    
    # 匹配每个实验的测试准确率
    TEST_ACC_PATTERN = re.compile(
        r"Loss\(test\)\s+[\d.]+\s*\|\s*Acc\(test\)\s+([\d.]+)",
        re.IGNORECASE
    )
    
    # 匹配最终统计中的最高训练准确率
    HIGHEST_TRAIN_PATTERN = re.compile(
        r"Highest\s+Train:\s*([\d.]+)\s*±\s*([\d.]+)",
        re.IGNORECASE
    )
    
    # 匹配最终统计中的最高验证准确率
    HIGHEST_VALID_PATTERN = re.compile(
        r"Highest\s+Valid:\s*([\d.]+)\s*±\s*([\d.]+)",
        re.IGNORECASE
    )
    
    # 匹配模型和数据集名称（从日志开头或文件名）
    MODEL_PATTERN = re.compile(r"模型[:\s]+(\w+)", re.IGNORECASE)
    DATASET_PATTERN = re.compile(r"数据集[:\s]+(\w+)", re.IGNORECASE)
    
    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.model: str = ""
        self.dataset: str = ""
        self.accuracy: Optional[float] = None  # Highest Train
        self.test_accuracy: Optional[float] = None  # 最后一个或平均的 Acc(test)
        self.final_test: Optional[float] = None  # 最后一个 Acc(test)
        self.valid_accuracy: Optional[float] = None  # Highest Valid
        self.test_accs: List[float] = []  # 所有测试准确率
        self.status: str = "UNKNOWN"
    
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        # 提取所有测试准确率
        test_acc_matches = list(self.TEST_ACC_PATTERN.finditer(self.log_content))
        for match in test_acc_matches:
            acc_value = float(match.group(1))
            # Acc(test) 是 0-1 之间的小数，转换为百分比
            if 0 <= acc_value <= 1:
                self.test_accs.append(acc_value * 100)
            else:
                # 已经是百分比格式
                self.test_accs.append(acc_value)
        
        # 计算测试准确率（取最后一个作为 final_test，平均值作为 test_accuracy）
        if self.test_accs:
            self.final_test = self.test_accs[-1]  # 最后一个
            self.test_accuracy = sum(self.test_accs) / len(self.test_accs)  # 平均值
        
        # 提取最高训练准确率
        train_match = self.HIGHEST_TRAIN_PATTERN.search(self.log_content)
        if train_match:
            self.accuracy = float(train_match.group(1))
        
        # 提取最高验证准确率
        valid_match = self.HIGHEST_VALID_PATTERN.search(self.log_content)
        if valid_match:
            self.valid_accuracy = float(valid_match.group(1))
        
        # 提取模型和数据集名称（如果还未提取）
        self._extract_model_dataset()
        
        # 状态：如果有测试准确率，则认为成功
        self.status = "OK" if (self.test_accuracy is not None or self.final_test is not None) else "FAILED"
        
        return {
            "model": self.model,
            "dataset": self.dataset,
            "accuracy": self.accuracy,
            "test_accuracy": self.test_accuracy,
            "final_test": self.final_test,
            "valid_accuracy": self.valid_accuracy,
            "test_accs": self.test_accs,
            "status": self.status
        }
    
    def _extract_model_dataset(self):
        """尝试从日志中提取模型和数据集名称"""
        # 提取模型名称
        if not self.model:
            model_match = self.MODEL_PATTERN.search(self.log_content)
            if model_match:
                self.model = model_match.group(1)
        
        # 提取数据集名称
        if not self.dataset:
            dataset_match = self.DATASET_PATTERN.search(self.log_content)
            if dataset_match:
                self.dataset = dataset_match.group(1)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取测试集指标（用于 ResultTable）"""
        metrics = {}
        if self.accuracy is not None:
            metrics['accuracy'] = self.accuracy
        if self.test_accuracy is not None:
            metrics['test_accuracy'] = self.test_accuracy
        if self.final_test is not None:
            metrics['final_test'] = self.final_test
        return metrics
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        获取最好的指标数据（输出格式：指标：数据值）
        
        返回:
            包含最好结果的字典，格式为 {指标名: 数据值}
        """
        best_metrics = {}
        if self.accuracy is not None:
            best_metrics['accuracy'] = self.accuracy
        if self.test_accuracy is not None:
            best_metrics['test_accuracy'] = self.test_accuracy
        if self.final_test is not None:
            best_metrics['final_test'] = self.final_test
        return best_metrics
    
    @classmethod
    def from_file(cls, log_path: Path, model: str = None, dataset: str = None) -> "GSLCollector":
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
    def from_string(cls, log_content: str, model: str = None, dataset: str = None) -> "GSLCollector":
        """从字符串读取"""
        collector = cls(log_content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector

