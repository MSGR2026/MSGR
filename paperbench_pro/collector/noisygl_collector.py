"""
NoisyGL 结果收集器

解析 NoisyGL 训练日志，提取指标

日志格式：
    Epoch 00001 | Time(s) 0.1234 | Loss(train) 0.5678 | Acc(train) 0.8234 | Loss(val) 0.4321 | Acc(val) 0.7890 | *
    ...
    Loss(test) 0.3456 | Acc(test) 0.8123
    Train Acc: 0.8234
    Valid Acc: 0.7890
    Test Acc: 0.8123

功能：
    - 提取所有训练过程中的验证准确率（每个 epoch 的验证指标）
    - 计算最佳验证准确率（取最大值）
    - 提取训练、验证、测试准确率
    - 提取训练过程中的损失值
"""
import re
from typing import Dict, Any, Optional, List
from pathlib import Path


class NoisyGLCollector:
    """NoisyGL 结果收集器"""
    
    # 正则匹配
    # 匹配 "Train Acc: value" 或 "Train Acc value"
    TRAIN_ACC_PATTERN = re.compile(
        r"Train\s+Acc\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配 "Valid Acc: value" 或 "Valid Acc value"
    VALID_ACC_PATTERN = re.compile(
        r"Valid\s+Acc\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配 "Test Acc: value" 或 "Test Acc value"
    TEST_ACC_PATTERN = re.compile(
        r"Test\s+Acc\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配 "Test Macro-F1: value"
    TEST_MACRO_F1_PATTERN = re.compile(
        r"Test\s+Macro-F1\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配 "Test Micro-F1: value"
    TEST_MICRO_F1_PATTERN = re.compile(
        r"Test\s+Micro-F1\s*:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配训练过程中的 epoch 输出：Acc(train) 和 Acc(val)
    # 格式：Epoch 00001 | Time(s) 0.1234 | Loss(train) 0.5678 | Acc(train) 0.8234 | Loss(val) 0.4321 | Acc(val) 0.7890 | *
    EPOCH_PATTERN = re.compile(
        r"Epoch\s+\d+\s*\|\s*Time\(s\)\s+[\d.]+\s*\|\s*Loss\(train\)\s+[\d.]+\s*\|\s*Acc\(train\)\s+([\d.]+)\s*\|\s*Loss\(val\)\s+[\d.]+\s*\|\s*Acc\(val\)\s+([\d.]+)",
        re.IGNORECASE
    )
    # 匹配测试输出：Acc(test)
    TEST_EPOCH_PATTERN = re.compile(
        r"Acc\(test\)\s+([\d.]+)",
        re.IGNORECASE
    )
    # 匹配模型和数据集名称
    MODEL_PATTERN = re.compile(
        r"Running\s+(\w+)\s+on\s+(\S+)",
        re.IGNORECASE
    )
    # 匹配噪声类型和噪声率
    NOISE_PATTERN = re.compile(
        r"Noise\s+type:\s*(\w+).*?Noise\s+rate:\s*([\d.]+)",
        re.IGNORECASE
    )
    # 匹配种子
    SEED_PATTERN = re.compile(
        r"Seed:\s*(\d+)",
        re.IGNORECASE
    )
    
    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.method: str = ""
        self.dataset: str = ""
        self.noise_type: str = ""
        self.noise_rate: Optional[float] = None
        self.seed: Optional[int] = None
        self.train_acc: Optional[float] = None
        self.valid_acc: Optional[float] = None
        self.test_acc: Optional[float] = None
        self.test_macro_f1: Optional[float] = None
        self.test_micro_f1: Optional[float] = None
        self.validation_accs: List[float] = []  # 存储所有验证准确率
        self.best_valid_acc: Optional[float] = None  # 最佳验证准确率
        self.train_accs: List[float] = []  # 存储所有训练准确率
        self.status: str = "UNKNOWN"
    
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        # 解析训练过程中的 epoch 输出
        epoch_matches = list(self.EPOCH_PATTERN.finditer(self.log_content))
        for match in epoch_matches:
            train_acc = float(match.group(1))
            valid_acc = float(match.group(2))
            self.train_accs.append(train_acc)
            self.validation_accs.append(valid_acc)
        
        # 计算最佳验证准确率（取最大值）
        if self.validation_accs:
            self.best_valid_acc = max(self.validation_accs)
        
        # 解析测试过程中的准确率
        test_epoch_matches = list(self.TEST_EPOCH_PATTERN.finditer(self.log_content))
        # 如果有多个匹配，取最后一个（通常是最终测试结果）
        
        # 解析最终摘要中的准确率（优先级更高）
        train_acc_match = self.TRAIN_ACC_PATTERN.search(self.log_content)
        if train_acc_match:
            self.train_acc = float(train_acc_match.group(1))
        
        valid_acc_match = self.VALID_ACC_PATTERN.search(self.log_content)
        if valid_acc_match:
            self.valid_acc = float(valid_acc_match.group(1))
        
        test_acc_match = self.TEST_ACC_PATTERN.search(self.log_content)
        if test_acc_match:
            self.test_acc = float(test_acc_match.group(1))
        
        # 解析F1分数
        test_macro_f1_match = self.TEST_MACRO_F1_PATTERN.search(self.log_content)
        if test_macro_f1_match:
            self.test_macro_f1 = float(test_macro_f1_match.group(1))
        
        test_micro_f1_match = self.TEST_MICRO_F1_PATTERN.search(self.log_content)
        if test_micro_f1_match:
            self.test_micro_f1 = float(test_micro_f1_match.group(1))
        
        # 如果没有从摘要中提取到，尝试从测试输出中提取
        if self.test_acc is None and test_epoch_matches:
            self.test_acc = float(test_epoch_matches[-1].group(1))
        
        # 提取模型、数据集、噪声信息
        self._extract_metadata()
        
        # 状态：如果有测试准确率，则认为成功
        self.status = "OK" if self.test_acc is not None else "FAILED"
        
        return {
            "method": self.method,
            "dataset": self.dataset,
            "noise_type": self.noise_type,
            "noise_rate": self.noise_rate,
            "seed": self.seed,
            "train_acc": self.train_acc,
            "valid_acc": self.valid_acc,
            "best_valid_acc": self.best_valid_acc,
            "validation_accs": self.validation_accs,  # 所有验证准确率
            "test_acc": self.test_acc,
            "test_macro_f1": self.test_macro_f1,
            "test_micro_f1": self.test_micro_f1,
            "train_accs": self.train_accs,  # 所有训练准确率
            "status": self.status
        }
    
    def _extract_metadata(self):
        """尝试从日志中提取元数据"""
        # 匹配 "Running METHOD on DATASET"
        model_match = self.MODEL_PATTERN.search(self.log_content)
        if model_match:
            self.method = model_match.group(1)
            self.dataset = model_match.group(2)
        
        # 匹配噪声类型和噪声率
        noise_match = self.NOISE_PATTERN.search(self.log_content)
        if noise_match:
            self.noise_type = noise_match.group(1)
            self.noise_rate = float(noise_match.group(2))
        
        # 匹配种子
        seed_match = self.SEED_PATTERN.search(self.log_content)
        if seed_match:
            self.seed = int(seed_match.group(1))
    
    def get_metrics(self) -> Dict[str, float]:
        """获取测试集指标（用于 ResultTable）"""
        metrics = {}
        if self.test_acc is not None:
            metrics['test_acc'] = self.test_acc
        if self.test_macro_f1 is not None:
            metrics['test_macro_f1'] = self.test_macro_f1
        if self.test_micro_f1 is not None:
            metrics['test_micro_f1'] = self.test_micro_f1
        if self.valid_acc is not None:
            metrics['valid_acc'] = self.valid_acc
        if self.train_acc is not None:
            metrics['train_acc'] = self.train_acc
        # 添加 best_valid_acc（从所有验证准确率中取最大值）
        if self.best_valid_acc is not None:
            metrics['best_valid_acc'] = self.best_valid_acc
        return metrics
    
    def get_validation_metrics(self) -> Dict[str, float]:
        """获取最佳验证指标"""
        metrics = {}
        if self.best_valid_acc is not None:
            metrics['valid_acc'] = self.best_valid_acc
        return metrics
    
    @classmethod
    def from_file(cls, log_path: Path, method: str = None, dataset: str = None) -> "NoisyGLCollector":
        """从文件读取"""
        content = Path(log_path).read_text(encoding="utf-8")
        collector = cls(content)
        collector.collect()
        if method:
            collector.method = method
        if dataset:
            collector.dataset = dataset
        return collector
    
    @classmethod
    def from_string(cls, log_content: str, method: str = None, dataset: str = None) -> "NoisyGLCollector":
        """从字符串读取"""
        collector = cls(log_content)
        collector.collect()
        if method:
            collector.method = method
        if dataset:
            collector.dataset = dataset
        return collector

