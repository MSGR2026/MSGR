"""
MMRec 结果收集器（实时解析）

逐行解析 MMRec 训练日志，只提取 Test 指标。

典型日志格式：
    INFO test result:
    recall@20: 0.0374 ... ndcg@20: 0.0172 ...
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path


class MMRecCollector:
    """MMRec 结果收集器（实时解析）"""

    # 指标匹配: recall@10: 0.1234 或 'recall@10': 0.1234
    METRIC_PATTERN = re.compile(r"['\"]?(\w+@\d+)['\"]?\s*:\s*([\d.]+)")

    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.model: str = ""
        self.dataset: str = ""
        self.best_valid: Dict[str, float] = {}
        self.test_result: Dict[str, float] = {}
        self.status: str = "UNKNOWN"
        self._await_test_metrics = False
        self._last_metrics_sig: Optional[tuple] = None

    def feed(self, line: str) -> Optional[Dict[str, float]]:
        """
        逐行解析日志。

        Returns:
            Test 指标字典（若该行触发新结果），否则 None。
        """
        stripped = line.strip()
        if not stripped:
            return None

        lower = stripped.lower()

        # 尝试提取模型和数据集名称
        if "model" in lower or "dataset" in lower:
            self._extract_model_dataset(stripped)

        # 检测 test/valid 边界
        if "test result" in lower or "best test" in lower:
            self._await_test_metrics = True
        if "valid result" in lower or "best valid" in lower:
            # 避免把 valid 的指标误认为 test
            self._await_test_metrics = False
            return None

        # 直接包含 Test: ... 的行或 test result 行本身带指标
        if "test:" in lower or "test result" in lower or "best test" in lower:
            metrics = self._extract_metrics(stripped)
            if metrics:
                self._await_test_metrics = False
                return self._update_test(metrics)

        # 在等待 test 指标时解析下一行
        if self._await_test_metrics:
            metrics = self._extract_metrics(stripped)
            if metrics:
                self._await_test_metrics = False
                return self._update_test(metrics)

        return None

    def collect(self) -> Dict[str, Any]:
        """收集指标（逐行解析 log_content）"""
        for line in self.log_content.splitlines():
            self.feed(line)

        self.status = "OK" if self.test_result else "FAILED"
        return {
            "model": self.model,
            "dataset": self.dataset,
            "best_valid": self.best_valid,
            "test_result": self.test_result,
            "status": self.status,
        }
    
    def _extract_metrics(self, content: str) -> Dict[str, float]:
        """提取指标"""
        metrics = {}
        for m in self.METRIC_PATTERN.finditer(content):
            name = m.group(1).lower()
            value = float(m.group(2))
            metrics[name] = value
        return metrics
    
    def _extract_model_dataset(self, text: str):
        """尝试从文本中提取模型和数据集名称"""
        model_match = re.search(r"model[:\s]+['\"]?(\w+)['\"]?", text, re.IGNORECASE)
        if model_match:
            self.model = model_match.group(1)

        dataset_match = re.search(r"dataset[:\s]+['\"]?(\w+)['\"]?", text, re.IGNORECASE)
        if dataset_match:
            self.dataset = dataset_match.group(1)

    def _update_test(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """更新 test 指标，去重后返回新结果"""
        if not metrics:
            return None
        sig = tuple(sorted(metrics.items()))
        if self._last_metrics_sig == sig:
            return None
        self._last_metrics_sig = sig
        self.test_result = metrics
        self.status = "OK"
        return metrics
    
    def get_metrics(self) -> Dict[str, float]:
        """获取测试集指标（用于 ResultTable）"""
        return self.test_result
    
    @classmethod
    def from_file(cls, log_path: Path, model: str = None, dataset: str = None) -> "MMRecCollector":
        """从文件读取（逐行解析）"""
        content = Path(log_path).read_text(encoding="utf-8")
        collector = cls(content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector

    @classmethod
    def from_string(cls, log_content: str, model: str = None, dataset: str = None) -> "MMRecCollector":
        """从字符串读取（逐行解析）"""
        collector = cls(log_content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector


def parse_mmrec_output(log_content: str) -> Dict[str, float]:
    """便捷函数：解析日志并返回指标"""
    collector = MMRecCollector.from_string(log_content)
    return collector.get_metrics()


# 测试
if __name__ == "__main__":
    # 测试格式1: 带花括号
    sample_log1 = """
    model: BM3
    dataset: baby
    ████Current BEST████:
    Parameters: learning_rate=0.001,
    Valid: {recall@10: 0.0567, recall@20: 0.0892, ndcg@10: 0.0312, ndcg@20: 0.0423},
    Test: {recall@10: 0.0612, recall@20: 0.0934, ndcg@10: 0.0345, ndcg@20: 0.0456}
    """
    
    print("=== 格式1: 带花括号 ===")
    collector = MMRecCollector.from_string(sample_log1)
    result = collector.collect()
    print(f"Model: {collector.model}")
    print(f"Dataset: {collector.dataset}")
    print(f"Status: {result['status']}")
    print(f"Metrics: {result['test_result']}")
    
    # 测试格式2: 无花括号 (实际 MMRec 输出格式)
    sample_log2 = """
    █████████████ BEST ████████████████
    31 Dec 21:20    INFO    Parameters: ['seed']=(999,),
    Valid: recall@5: 0.0187    recall@10: 0.0292    recall@20: 0.0482    ndcg@10: 0.0161    ndcg@20: 0.0209    ,
    Test: recall@5: 0.0192    recall@10: 0.0301    recall@20: 0.0484    ndcg@10: 0.0162    ndcg@20: 0.0209    
    """
    
    print("\n=== 格式2: 无花括号 ===")
    collector2 = MMRecCollector.from_string(sample_log2)
    result2 = collector2.collect()
    print(f"Status: {result2['status']}")
    print(f"Best Valid: {result2['best_valid']}")
    print(f"Test Metrics: {result2['test_result']}")
