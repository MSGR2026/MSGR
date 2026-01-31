"""
GeneralGL 结果收集器

解析 GeneralGL 训练日志，提取指标

日志格式：
    train: {'epoch': 0, 'time_epoch': 17.98859, 'loss': 6.2047755, 'accuracy': 0.00146, 'f1': 0.00069, 'micro-f1': 0.00146}
    val: {'epoch': 0, 'time_epoch': 22.97002, 'loss': 6.09097182, 'accuracy': 0.0014, 'f1': 0.00029, 'micro-f1': 0.0014}
    test: {'epoch': 0, 'time_epoch': 23.21948, 'loss': 6.07986232, 'accuracy': 0.0011, 'f1': 0.00058, 'micro-f1': 0.0011}
    > Epoch 0: took 67.5s (avg 67.5s) | Best so far: epoch 0	train_loss: 6.2048 train_accuracy: 0.0015	val_loss: 6.0910 val_accuracy: 0.0014	test_loss: 6.0799 test_accuracy: 0.0011

功能：
    - 提取所有验证结果（每个 epoch 的验证指标）
    - 计算最佳验证结果（根据 metric_best 配置，通常是 accuracy 或 f1）
    - 提取测试结果
    - 提取训练、验证、测试指标（accuracy, f1, micro-f1 等）
"""
import re
import ast
from typing import Dict, Any, Optional, List
from pathlib import Path


class GeneralGLCollector:
    """GeneralGL 结果收集器"""
    
    # 正则匹配
    # 匹配 train/val/test 字典格式的输出
    TRAIN_DICT_PATTERN = re.compile(
        r"^train:\s*(\{.*?\})",
        re.MULTILINE
    )
    VAL_DICT_PATTERN = re.compile(
        r"^val:\s*(\{.*?\})",
        re.MULTILINE
    )
    TEST_DICT_PATTERN = re.compile(
        r"^test:\s*(\{.*?\})",
        re.MULTILINE
    )
    # 匹配 "Best so far" 行
    BEST_EPOCH_PATTERN = re.compile(
        r">\s*Epoch\s+(\d+).*?Best\s+so\s+far:\s+epoch\s+(\d+)\s+train_loss:\s+([\d.]+)\s+train_accuracy:\s+([\d.]+)\s+val_loss:\s+([\d.]+)\s+val_accuracy:\s+([\d.]+)\s+test_loss:\s+([\d.]+)\s+test_accuracy:\s+([\d.]+)",
        re.IGNORECASE
    )
    # 匹配 metric_best 配置
    METRIC_BEST_PATTERN = re.compile(
        r"metric_best:\s*(\w+)",
        re.IGNORECASE
    )
    # 匹配 "Best Test Metrics" 和 "Best Validation Metrics" 字典格式（从 Collector Results 部分）
    BEST_TEST_METRICS_PATTERN = re.compile(
        r"Best\s+Test\s+Metrics:\s*(\{.*?\})",
        re.IGNORECASE | re.DOTALL
    )
    BEST_VALIDATION_METRICS_PATTERN = re.compile(
        r"Best\s+Validation\s+Metrics:\s*(\{.*?\})",
        re.IGNORECASE | re.DOTALL
    )
    # 匹配数据集和模型名称
    DATASET_PATTERN = re.compile(
        r"Loaded\s+dataset\s+['\"](\S+)['\"]|Dataset:\s*(\S+)",
        re.IGNORECASE
    )
    MODEL_PATTERN = re.compile(
        r"name_tag[:\s=]+['\"]?(\S+)['\"]?|Model:\s*(\S+)",
        re.IGNORECASE
    )
    RUN_ID_PATTERN = re.compile(
        r"Run\s+ID\s+(\d+)",
        re.IGNORECASE
    )
    SEED_PATTERN = re.compile(
        r"seed[=:]?\s*(\d+)",
        re.IGNORECASE
    )
    
    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.model: str = ""
        self.dataset: str = ""
        self.run_id: Optional[int] = None
        self.seed: Optional[int] = None
        self.metric_best: str = "accuracy"  # 默认使用 accuracy
        self.validation_metrics: Dict[str, List[float]] = {}  # 存储所有验证结果
        self.best_validation_metrics: Dict[str, float] = {}  # 存储最佳验证结果
        self.test_metrics: Dict[str, float] = {}
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.test_metrics_list: List[Dict[str, float]] = []
        self.best_epoch: Optional[int] = None
        self.best_test_metrics: Dict[str, float] = {}
        self.status: str = "UNKNOWN"
    
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        # 首先尝试从 "Collector Results" 部分提取（如果存在）
        best_test_match = self.BEST_TEST_METRICS_PATTERN.search(self.log_content)
        if best_test_match:
            try:
                metrics_dict = ast.literal_eval(best_test_match.group(1))
                if isinstance(metrics_dict, dict):
                    # 将键转换为小写以保持一致性
                    self.best_test_metrics = {k.lower(): v for k, v in metrics_dict.items()}
            except (ValueError, SyntaxError):
                pass
        
        best_val_match = self.BEST_VALIDATION_METRICS_PATTERN.search(self.log_content)
        if best_val_match:
            try:
                metrics_dict = ast.literal_eval(best_val_match.group(1))
                if isinstance(metrics_dict, dict):
                    # 将键转换为小写以保持一致性
                    self.best_validation_metrics = {k.lower(): v for k, v in metrics_dict.items()}
            except (ValueError, SyntaxError):
                pass
        
        # 如果已经从 Collector Results 中提取到指标，直接返回
        if self.best_test_metrics:
            self.status = "OK"
            self._extract_model_dataset()
            return {
                "model": self.model,
                "dataset": self.dataset,
                "run_id": self.run_id,
                "seed": self.seed,
                "metric_best": self.metric_best,
                "best_epoch": self.best_epoch,
                "validation_metrics": self.validation_metrics,
                "best_validation_metrics": self.best_validation_metrics,
                "test_metrics": self.best_test_metrics,
                "best_test_metrics": self.best_test_metrics,
                "status": self.status
            }
        
        # 否则，从原始训练日志中解析
        # 提取 metric_best 配置
        metric_match = self.METRIC_BEST_PATTERN.search(self.log_content)
        if metric_match:
            self.metric_best = metric_match.group(1).lower()
        
        # 解析训练过程中的字典输出
        train_matches = self.TRAIN_DICT_PATTERN.finditer(self.log_content)
        for match in train_matches:
            try:
                metrics_dict = ast.literal_eval(match.group(1))
                if isinstance(metrics_dict, dict):
                    self.train_metrics.append(metrics_dict)
            except (ValueError, SyntaxError):
                pass
        
        # 解析验证指标（所有 epoch 的验证结果）
        val_matches = self.VAL_DICT_PATTERN.finditer(self.log_content)
        for match in val_matches:
            try:
                metrics_dict = ast.literal_eval(match.group(1))
                if isinstance(metrics_dict, dict):
                    self.val_metrics.append(metrics_dict)
                    # 提取验证指标到 validation_metrics 字典中
                    for metric_name, metric_value in metrics_dict.items():
                        if metric_name not in ['epoch', 'time_epoch', 'time_iter', 'lr', 'params', 'eta', 'eta_hours']:
                            metric_key = metric_name.upper()
                            if metric_key not in self.validation_metrics:
                                self.validation_metrics[metric_key] = []
                            if isinstance(metric_value, (int, float)):
                                self.validation_metrics[metric_key].append(float(metric_value))
            except (ValueError, SyntaxError):
                pass
        
        # 计算最佳验证结果（根据 metric_best 配置）
        metric_best_key = self.metric_best.upper()
        if metric_best_key in self.validation_metrics and self.validation_metrics[metric_best_key]:
            # 取最大值（适用于 accuracy, f1 等指标）
            self.best_validation_metrics[metric_best_key] = max(self.validation_metrics[metric_best_key])
            # 找到最佳 epoch
            best_value = self.best_validation_metrics[metric_best_key]
            for val_metric in self.val_metrics:
                if val_metric.get(self.metric_best) == best_value:
                    self.best_epoch = val_metric.get('epoch')
                    break
        
        # 解析测试指标
        test_matches = self.TEST_DICT_PATTERN.finditer(self.log_content)
        for match in test_matches:
            try:
                metrics_dict = ast.literal_eval(match.group(1))
                if isinstance(metrics_dict, dict):
                    self.test_metrics_list.append(metrics_dict)
            except (ValueError, SyntaxError):
                pass
        
        # 解析 "Best so far" 行，获取最佳 epoch 的测试结果
        best_matches = list(self.BEST_EPOCH_PATTERN.finditer(self.log_content))
        if best_matches:
            # 取最后一个匹配（通常是最终的最佳结果）
            last_match = best_matches[-1]
            self.best_epoch = int(last_match.group(2))  # Best so far epoch
            
            # 提取最佳测试指标
            if not self.best_test_metrics:
                self.best_test_metrics = {
                    'loss': float(last_match.group(7)),
                    'accuracy': float(last_match.group(8))
                }
            
            # 尝试从对应 epoch 的字典中获取更多指标（如 f1, micro-f1）
            for test_metric in self.test_metrics_list:
                if test_metric.get('epoch') == self.best_epoch:
                    self.best_test_metrics.update({
                        k: v for k, v in test_metric.items() 
                        if k not in ['epoch', 'time_epoch', 'time_iter', 'lr', 'params']
                    })
                    break
            
            # 提取最佳验证指标（从 "Best so far" 行）
            if not self.best_validation_metrics:
                self.best_validation_metrics.update({
                    'loss': float(last_match.group(5)),
                    'accuracy': float(last_match.group(6))
                })
            
            # 尝试从对应 epoch 的验证字典中获取更多指标
            for val_metric in self.val_metrics:
                if val_metric.get('epoch') == self.best_epoch:
                    self.best_validation_metrics.update({
                        k: v for k, v in val_metric.items() 
                        if k not in ['epoch', 'time_epoch', 'time_iter', 'lr', 'params', 'eta', 'eta_hours']
                    })
                    break
        
        # 如果没有找到最佳 epoch，使用最后一个验证结果
        if self.best_epoch is None and self.val_metrics:
            last_val = self.val_metrics[-1]
            self.best_epoch = last_val.get('epoch', len(self.val_metrics) - 1)
            if not self.best_validation_metrics:
                self.best_validation_metrics = {
                    k: v for k, v in last_val.items() 
                    if k not in ['epoch', 'time_epoch', 'time_iter', 'lr', 'params', 'eta', 'eta_hours']
                }
            # 尝试找到对应的测试结果
            for test_metric in self.test_metrics_list:
                if test_metric.get('epoch') == self.best_epoch:
                    if not self.best_test_metrics:
                        self.best_test_metrics = {
                            k: v for k, v in test_metric.items() 
                            if k not in ['epoch', 'time_epoch', 'time_iter', 'lr', 'params']
                        }
                    break
        
        # 提取测试指标（用于兼容性，使用最佳测试指标）
        self.test_metrics = self.best_test_metrics.copy()
        
        # 提取元数据
        self._extract_model_dataset()
        
        # 状态：如果有测试指标，则认为成功
        self.status = "OK" if self.best_test_metrics else "FAILED"
        
        return {
            "model": self.model,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "seed": self.seed,
            "metric_best": self.metric_best,
            "best_epoch": self.best_epoch,
            "validation_metrics": self.validation_metrics,  # 所有验证结果
            "best_validation_metrics": self.best_validation_metrics,  # 最佳验证结果
            "test_metrics": self.test_metrics,
            "best_test_metrics": self.best_test_metrics,
            "status": self.status
        }
    
    def _extract_model_dataset(self):
        """尝试从日志中提取模型和数据集名称"""
        # 匹配数据集名称
        dataset_match = self.DATASET_PATTERN.search(self.log_content)
        if dataset_match:
            self.dataset = dataset_match.group(1) or dataset_match.group(2)
        
        # 匹配模型名称（从 name_tag 或 Model:）
        model_match = self.MODEL_PATTERN.search(self.log_content)
        if model_match:
            self.model = model_match.group(1) or model_match.group(2)
        
        # 匹配 Run ID
        run_id_match = self.RUN_ID_PATTERN.search(self.log_content)
        if run_id_match:
            self.run_id = int(run_id_match.group(1))
        
        # 匹配种子
        seed_match = self.SEED_PATTERN.search(self.log_content)
        if seed_match:
            self.seed = int(seed_match.group(1))
    
    def get_metrics(self) -> Dict[str, float]:
        """获取测试集指标（用于 ResultTable）"""
        return self.best_test_metrics
    
    def get_validation_metrics(self) -> Dict[str, float]:
        """获取最佳验证指标"""
        return self.best_validation_metrics
    
    @classmethod
    def from_file(cls, log_path: Path, model: str = None, dataset: str = None) -> "GeneralGLCollector":
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
    def from_string(cls, log_content: str, model: str = None, dataset: str = None) -> "GeneralGLCollector":
        """从字符串读取"""
        collector = cls(log_content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector

