"""
TimeSeries 结果收集器（实时解析）

解析 TimeSeries 训练日志，支持实时逐行解析和批量解析。

典型日志格式：
    长期预测: mse:0.123, mae:0.456, dtw:xxx
    短期预测: smape: {'Average': 14.591, 'Yearly': 13.564, ...}
    分类任务: accuracy:0.789
    训练过程: Epoch: 1, Steps: 100 | Train Loss: 0.123 Vali Loss: 0.456 Test Acc: 0.789
"""
import re
from typing import Dict, Any, Optional
from pathlib import Path


class TimeSeriesCollector:
    """TimeSeries 结果收集器（实时解析）"""
    
    # 预测任务指标: mse:0.123, mae:0.456, rmse:0.789
    FORECAST_PATTERN = re.compile(
        r"(mse|mae|rmse|mape|mspe|dtw)\s*[:=]\s*([\d.]+|'Not calculated'|Not calculated)",
        re.IGNORECASE
    )
    
    # 分类任务指标: accuracy:0.789 或 Accuracy: 0.789
    CLASSIFICATION_PATTERN = re.compile(
        r"(accuracy|precision|recall|f1|f-score)\s*[:=]\s*([\d.]+)",
        re.IGNORECASE
    )
    
    # 短期预测字典格式: smape: {'Average': 14.591, 'Yearly': 13.564, ...}
    # 注意：这是关键差异点！
    DICT_PATTERN = re.compile(
        r"(smape|mape|mase|owa)\s*:\s*(\{[^}]+\})",
        re.IGNORECASE
    )
    
    # 训练过程中的测试指标: Test Acc: 0.789 或 Test Loss: 0.123
    TRAIN_TEST_PATTERN = re.compile(
        r"Test\s+(Acc|Loss|Accuracy)\s*[:=]\s*([\d.]+)",
        re.IGNORECASE
    )
    
    # Epoch 行: Epoch: 1, Steps: 100 | Train Loss: 0.123 ...
    EPOCH_PATTERN = re.compile(
        r"Epoch:\s*(\d+)",
        re.IGNORECASE
    )
    
    def __init__(self, log_content: str = "", task: str = "long_term_forecast"):
        self.log_content = log_content
        self.task = task
        self.test_metrics: Dict[str, float] = {}  # 最终测试指标
        self.train_metrics: Dict[str, float] = {}  # 训练过程中的最佳指标
        self.status: str = "UNKNOWN"
        self._last_epoch: Optional[int] = None
        self._last_test_sig: Optional[tuple] = None
        self._in_test_phase = False  # 是否进入测试阶段
        
        # 短期预测特有：收集所有4个指标的字典
        self._short_term_dicts: Dict[str, Dict[str, float]] = {}
    
    def feed(self, line: str) -> Optional[Dict[str, float]]:
        """
        逐行解析日志（实时收集）
        
        Returns:
            指标字典（若该行触发新结果），否则 None
        """
        stripped = line.strip()
        if not stripped:
            return None
        
        lower = stripped.lower()
        
        # 检测测试阶段开始
        if 'testing' in lower or 'test shape:' in lower:
            self._in_test_phase = True
            return None
        
        # 1. 尝试解析最终测试结果（优先级最高）
        if self._in_test_phase:
            metrics = self._extract_final_test_metrics(stripped)
            if metrics:
                return self._update_test(metrics)
        
        # 2. 解析训练过程中的测试指标（用于实时显示）
        if 'epoch:' in lower:
            epoch_match = self.EPOCH_PATTERN.search(stripped)
            if epoch_match:
                self._last_epoch = int(epoch_match.group(1))
            
            # 提取训练过程中的测试指标
            train_test_metrics = self._extract_train_test_metrics(stripped)
            if train_test_metrics:
                # 训练过程中的指标作为临时最佳结果
                return self._update_train(train_test_metrics)
        
        return None
    
    def collect(self) -> Dict[str, Any]:
        """收集指标（逐行解析 log_content）"""
        for line in self.log_content.splitlines():
            self.feed(line)
        
        # 优先使用最终测试指标，如果没有则使用训练过程中的最佳指标
        final_metrics = self.test_metrics if self.test_metrics else self.train_metrics
        self.status = "OK" if final_metrics else "FAILED"
        
        return {
            "task": self.task,
            "metrics": final_metrics,
            "status": self.status
        }
    
    def _extract_final_test_metrics(self, content: str) -> Dict[str, float]:
        """提取最终测试指标（test() 函数输出）"""
        metrics = {}
        
        # 1. 短期预测的字典格式（关键差异！）
        # 示例: smape: {'Average': 14.591, 'Yearly': 13.564, ...}
        for match in self.DICT_PATTERN.finditer(content):
            key, dict_str = match.groups()
            try:
                import ast
                metrics_dict = ast.literal_eval(dict_str)
                if isinstance(metrics_dict, dict):
                    # 存储完整字典供调试
                    self._short_term_dicts[key.lower()] = metrics_dict
                    
                    # 提取 Average 作为主要指标
                    if 'Average' in metrics_dict:
                        metrics[key.lower()] = float(metrics_dict['Average'])
            except Exception:
                # 解析失败时记录但不中断
                pass
        
        # 2. 根据任务类型解析标准格式
        if self.task in ["long_term_forecast", "imputation", "anomaly_detection"]:
            # 长期预测: mse:0.123, mae:0.456
            for match in self.FORECAST_PATTERN.finditer(content):
                key, value = match.groups()
                if value not in ["'Not calculated'", "Not calculated"]:
                    try:
                        metrics[key.lower()] = float(value)
                    except ValueError:
                        pass
        
        elif self.task == "short_term_forecast":
            # 短期预测已在上面的字典解析中处理
            # 这里可以添加额外的标量指标（如果有）
            for match in self.FORECAST_PATTERN.finditer(content):
                key, value = match.groups()
                if value not in ["'Not calculated'", "Not calculated"] and key.lower() not in metrics:
                    try:
                        metrics[key.lower()] = float(value)
                    except ValueError:
                        pass
        
        elif self.task == "classification":
            # 分类: accuracy:0.789
            for match in self.CLASSIFICATION_PATTERN.finditer(content):
                key, value = match.groups()
                metrics[key.lower()] = float(value)
        
        return metrics
    
    def _extract_train_test_metrics(self, content: str) -> Dict[str, float]:
        """提取训练过程中的测试指标（Epoch 行中的 Test Acc/Loss）"""
        metrics = {}
        
        # 提取 Test Acc 或 Test Loss
        for match in self.TRAIN_TEST_PATTERN.finditer(content):
            metric_name, value = match.groups()
            # 统一命名：Test Acc -> accuracy, Test Loss -> test_loss
            if metric_name.lower() in ['acc', 'accuracy']:
                metrics['accuracy'] = float(value)
            elif metric_name.lower() == 'loss':
                metrics['test_loss'] = float(value)
        
        return metrics
    
    def _update_test(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """更新最终测试指标，去重后返回新结果"""
        if not metrics:
            return None
        
        sig = tuple(sorted(metrics.items()))
        if self._last_test_sig == sig:
            return None
        
        self._last_test_sig = sig
        self.test_metrics = metrics
        self.status = "OK"
        return metrics
    
    def _update_train(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """更新训练过程中的指标（用于实时显示）"""
        if not metrics:
            return None
        
        # 只有当指标更好时才更新（对于 accuracy 越大越好，对于 loss 越小越好）
        should_update = False
        
        if 'accuracy' in metrics:
            old_acc = self.train_metrics.get('accuracy', 0)
            if metrics['accuracy'] > old_acc:
                should_update = True
        elif 'test_loss' in metrics:
            old_loss = self.train_metrics.get('test_loss', float('inf'))
            if metrics['test_loss'] < old_loss:
                should_update = True
        
        if should_update:
            self.train_metrics.update(metrics)
            return metrics
        
        return None
    
    def get_metrics(self) -> Dict[str, float]:
        """获取指标（用于 ResultTable）"""
        if not self.test_metrics and not self.train_metrics:
            self.collect()
        # 优先返回最终测试指标
        return self.test_metrics if self.test_metrics else self.train_metrics
    
    def get_short_term_details(self) -> Dict[str, Dict[str, float]]:
        """获取短期预测的详细分组指标（可选）"""
        return self._short_term_dicts
    
    @classmethod
    def from_string(cls, log_content: str, task: str = "long_term_forecast") -> "TimeSeriesCollector":
        """从字符串读取"""
        collector = cls(log_content, task)
        collector.collect()
        return collector
    
    @classmethod
    def from_file(cls, log_path: Path, task: str = "long_term_forecast") -> "TimeSeriesCollector":
        """从文件读取"""
        content = Path(log_path).read_text(encoding="utf-8")
        return cls.from_string(content, task)


def parse_timeseries_output(log_content: str, task: str = "long_term_forecast") -> Dict[str, float]:
    """便捷函数：解析日志并返回指标"""
    collector = TimeSeriesCollector.from_string(log_content, task)
    return collector.get_metrics()


# 测试
if __name__ == "__main__":
    # 测试短期预测（关键差异！）
    sample_log_short_term = """
    >>>>>>>testing : m4_Yearly_DLinear_custom_ftM_sl24_ll12_pl6_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    test shape: (23000, 6, 1)
    DLinear
    smape: {'Average': 14.591, 'Yearly': 13.564, 'Quarterly': 10.156, 'Monthly': 12.345, 'Others': 15.678}
    mape: {'Average': 12.345, 'Yearly': 11.234, 'Quarterly': 9.876, 'Monthly': 11.567, 'Others': 13.456}
    mase: {'Average': 3.456, 'Yearly': 3.123, 'Quarterly': 2.987, 'Monthly': 3.234, 'Others': 3.789}
    owa: {'Average': 0.987, 'Yearly': 0.923, 'Quarterly': 0.876, 'Monthly': 0.945, 'Others': 1.023}
    """
    
    print("=== 测试短期预测（字典格式）===")
    collector = TimeSeriesCollector(task="short_term_forecast")
    for line in sample_log_short_term.splitlines():
        result = collector.feed(line)
        if result:
            print(f"实时更新: {result}")
    
    print(f"\n最终指标: {collector.get_metrics()}")
    print(f"详细分组: {collector.get_short_term_details()}")
    print(f"状态: {collector.status}")
    
    # 测试长期预测
    sample_log_forecast = """
    >>>>>>>testing : weather_96_96_Autoformer<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    test shape: (10444, 96, 21) (10444, 96, 21)
    mse:0.2845, mae:0.3234, dtw:Not calculated
    """
    
    print("\n=== 测试长期预测 ===")
    collector2 = TimeSeriesCollector.from_string(sample_log_forecast, task="long_term_forecast")
    print(f"最终指标: {collector2.get_metrics()}")
    
    # 测试分类任务
    sample_log_classification = """
    Epoch: 1, Steps: 50 | Train Loss: 1.234 Vali Loss: 1.123 Vali Acc: 0.456 Test Loss: 1.089 Test Acc: 0.478
    Epoch: 2, Steps: 50 | Train Loss: 0.987 Vali Loss: 0.876 Vali Acc: 0.567 Test Loss: 0.845 Test Acc: 0.589
    Epoch: 3, Steps: 50 | Train Loss: 0.765 Vali Loss: 0.654 Vali Acc: 0.678 Test Loss: 0.623 Test Acc: 0.701
    >>>>>>>testing : EthanolConcentration_PatchTST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    test shape: torch.Size([500, 5]) torch.Size([500])
    accuracy:0.7234
    """
    
    print("\n=== 测试分类任务（实时解析）===")
    collector3 = TimeSeriesCollector(task="classification")
    for line in sample_log_classification.splitlines():
        result = collector3.feed(line)
        if result:
            print(f"实时更新: {result}")
    
    print(f"\n最终指标: {collector3.get_metrics()}")
    print(f"状态: {collector3.status}")
