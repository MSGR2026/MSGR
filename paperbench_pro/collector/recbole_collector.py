"""
RecBole 结果收集器

解析 RecBole 训练日志，提取指标


"""
import re
from typing import Dict, Any, Optional
from pathlib import Path


class RecBoleCollector:
    """RecBole 结果收集器"""
    
    # 正则匹配
    TEST_RESULT_PATTERN = re.compile(
        r"^.*?\btest\s+result\s*:\s*(.+)$",
        re.IGNORECASE | re.MULTILINE
    )
    BEST_VALID_PATTERN = re.compile(
        r"^.*?\bbest\s+valid\s*:\s*(.+)$",
        re.IGNORECASE | re.MULTILINE
    )
    VALUE_PATTERN = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    # 修改第三个分组：要求 key 必须包含字母或 @，避免匹配纯数字（如时间戳 16:34）
    METRIC_PATTERN = re.compile(
        rf"(?:\(\s*['\"](?P<key1>[^'\"]+)['\"]\s*,\s*(?P<val1>{VALUE_PATTERN})\s*\))"
        rf"|(?:['\"](?P<key2>[^'\"]+)['\"]\s*:\s*(?P<val2>{VALUE_PATTERN}))"
        rf"|(?:\b(?P<key3>[A-Za-z_][A-Za-z0-9_]*(?:@\d+)?)\s*:\s*(?P<val3>{VALUE_PATTERN}))"
    )
    
    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.model: str = ""
        self.dataset: str = ""
        self.best_valid: Dict[str, float] = {}
        self.test_result: Dict[str, float] = {}
        self.status: str = "UNKNOWN"
        self._await_test_metrics = False
        self._await_valid_metrics = False
        self._last_metrics_sig: Optional[tuple] = None

    def feed(self, line: str) -> Optional[Dict[str, float]]:
        """
        逐行解析日志，返回测试集指标（若该行触发新结果）
        """
        stripped = line.strip()
        if not stripped:
            return None

        lower = stripped.lower()

        if "model" in lower or "dataset" in lower:
            self._extract_model_dataset(stripped)

        if "test result" in lower or "best test" in lower:
            self._await_test_metrics = True
        if "best valid" in lower or "valid result" in lower:
            self._await_valid_metrics = True

        if "test result" in lower or "best test" in lower:
            metrics = self._extract_metrics(stripped)
            if metrics:
                self._await_test_metrics = False
                return self._update_test(metrics)

        if "best valid" in lower or "valid result" in lower:
            metrics = self._extract_metrics(stripped)
            if metrics:
                self._await_valid_metrics = False
                self.best_valid = metrics

        if self._await_test_metrics:
            metrics = self._extract_metrics(stripped)
            if metrics:
                self._await_test_metrics = False
                return self._update_test(metrics)

        if self._await_valid_metrics:
            metrics = self._extract_metrics(stripped)
            if metrics:
                self._await_valid_metrics = False
                self.best_valid = metrics

        return None
    
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        for line in self.log_content.splitlines():
            self.feed(line)
        
        # 提取模型和数据集名称
        self._extract_model_dataset()
        
        # 状态
        self.status = "OK" if self.test_result else "FAILED"
        
        return {
            "model": self.model,
            "dataset": self.dataset,
            "best_valid": self.best_valid,
            "test_result": self.test_result,
            "status": self.status
        }
    
    def _extract_metrics(self, content: str) -> Dict[str, float]:
        """提取指标"""
        metrics = {}
        for m in self.METRIC_PATTERN.finditer(content):
            name = m.group("key1") or m.group("key2") or m.group("key3")
            value = m.group("val1") or m.group("val2") or m.group("val3")
            if name is None or value is None:
                continue
            metrics[name.lower()] = float(value)
        return metrics
    
    def _extract_model_dataset(self, text: Optional[str] = None):
        """尝试从日志中提取模型和数据集名称"""
        content = text if text is not None else self.log_content
        # 匹配 model: xxx 或 dataset: xxx
        model_match = re.search(r"model[:\s]+['\"]?(\w+)['\"]?", content, re.IGNORECASE)
        if model_match:
            self.model = model_match.group(1)
        
        dataset_match = re.search(r"dataset[:\s]+['\"]?(\w+)['\"]?", content, re.IGNORECASE)
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
    def from_file(cls, log_path: Path, model: str = None, dataset: str = None) -> "RecBoleCollector":
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
    def from_string(cls, log_content: str, model: str = None, dataset: str = None) -> "RecBoleCollector":
        """从字符串读取"""
        collector = cls(log_content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector


def parse_recbole_output(log_content: str) -> Dict[str, float]:
    """便捷函数：解析日志并返回指标"""
    collector = RecBoleCollector.from_string(log_content)
    return collector.get_metrics()


# 测试
if __name__ == "__main__":
    sample_log = """
    recall@10 : 0.2108    mrr@10 : 0.3809    ndcg@10 : 0.2298    hit@10 : 0.7455    precision@10 : 0.1618
    Fri 02 Jan 2026 19:49:50 INFO  epoch 78 training [time: 0.67s, train loss: 57544.8965]
    Fri 02 Jan 2026 19:49:50 INFO  epoch 78 evaluating [time: 0.40s, valid_score: 0.382900]
    Fri 02 Jan 2026 19:49:50 INFO  valid result: 
    recall@10 : 0.2108    mrr@10 : 0.3829    ndcg@10 : 0.23    hit@10 : 0.7444    precision@10 : 0.1619
    Fri 02 Jan 2026 19:49:50 INFO  Finished training, best eval result in epoch 67
    Fri 02 Jan 2026 19:49:50 INFO  Loading model structure and parameters from saved/SGL-Jan-02-2026_19-48-15.pth
    Fri 02 Jan 2026 19:49:52 INFO  The running environment of this training is as follows:
    +-------------+-----------------+
    | Environment |      Usage      |
    +=============+=================+
    | CPU         |     11.00 %     |
    +-------------+-----------------+
    | GPU         | 0.12 G/79.14 G  |
    +-------------+-----------------+
    | Memory      | 1.32 G/251.40 G |
    +-------------+-----------------+
    Fri 02 Jan 2026 19:49:52 INFO  best valid : OrderedDict({'recall@20': 0.2078, 'mrr@20': 0.3841, 'ndcg@20': 0.2287, 'hit@20': 0.7497, 'precision@20': 0.1603})
    Fri 02 Jan 2026 19:49:52 INFO  test result: OrderedDict({'recall@20': 0.2487, 'mrr@20': 0.4837, 'ndcg@20': 0.2907, 'hit@20': 0.7879, 'precision@20': 0.1993})
    """
    
    collector = RecBoleCollector.from_string(sample_log)
    result = collector.collect()
    print(f"Model: {collector.model}")
    print(f"Dataset: {collector.dataset}")
    print(f"Status: {result['status']}")
    print(f"Metrics: {result['test_result']}")
