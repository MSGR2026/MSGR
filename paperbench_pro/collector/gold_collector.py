"""
GOLD 结果收集器

解析 GOLD Benchmark 训练日志，提取指标

日志格式：
    每个试验的结果：
        AUROC: 0.44362564480471633
        AUPRC: 0.358378867891578
        FPR95: 0.9891304347826086
    
    最终结果统计：
        ============================================================
        最终结果统计
        ============================================================
        数据集: DHFR
        AUROC: 52.69% (方差: 0.34%)
        AUPRC: 43.25% (方差: 0.34%)
        FPR95: ... (方差: ...)  # 可能缺失
"""
import re
from typing import Dict, Any, Optional, List
from pathlib import Path


class GOLDCollector:
    """GOLD Benchmark 结果收集器"""
    
    # 正则匹配最终结果统计（多行格式，FPR95 可选）
    # 支持数据集对格式（如 AIDS+DHFR, BZR+COX2）
    # 支持带时间戳的日志格式（如 23:31:00 数据集: AIDS+DHFR）
    FINAL_STATS_PATTERN = re.compile(
        r"最终结果统计\s*\n"
        r"[^\n]*=+\s*\n"  # 可能包含时间戳的分隔线
        r"(?:[^\n]*\s+)?数据集:\s*([\w+]+)\s*\n"  # 可能包含时间戳
        r"(?:[^\n]*\s+)?AUROC:\s*([\d.]+)%\s*\(方差:\s*([\d.]+)%\)\s*\n"  # 可能包含时间戳
        r"(?:[^\n]*\s+)?AUPRC:\s*([\d.]+)%\s*\(方差:\s*([\d.]+)%\)"  # 可能包含时间戳
        r"(?:\s*\n(?:[^\n]*\s+)?FPR95:\s*([\d.]+)%\s*\(方差:\s*([\d.]+)%\))?",  # FPR95 可选，可能包含时间戳
        re.IGNORECASE | re.MULTILINE
    )
    
    # 匹配每个试验的指标（用于计算平均值，如果最终统计缺失）
    TRIAL_AUROC_PATTERN = re.compile(r"^AUROC:\s*([\d.]+)$", re.MULTILINE)
    TRIAL_AUPRC_PATTERN = re.compile(r"^AUPRC:\s*([\d.]+)$", re.MULTILINE)
    TRIAL_FPR95_PATTERN = re.compile(r"^FPR95:\s*([\d.]+)$", re.MULTILINE)
    
    # 匹配模型和数据集名称（从日志开头或文件名）
    # 支持数据集对格式（如 AIDS+DHFR）
    MODEL_PATTERN = re.compile(r"模型[:\s]+(\w+)", re.IGNORECASE)
    DATASET_PATTERN = re.compile(r"数据集[:\s]+([\w+]+)", re.IGNORECASE)
    
    def __init__(self, log_content: str = ""):
        self.log_content = log_content
        self.model: str = ""
        self.dataset: str = ""
        self.auroc: float = 0.0
        self.auprc: float = 0.0
        self.fpr95: float = 0.0
        self.auroc_std: float = 0.0
        self.auprc_std: float = 0.0
        self.fpr95_std: float = 0.0
        self.status: str = "UNKNOWN"
        self.trial_results: List[Dict[str, float]] = []
    
    def collect(self) -> Dict[str, Any]:
        """收集指标"""
        # 优先解析最终结果统计
        match = self.FINAL_STATS_PATTERN.search(self.log_content)
        if match:
            self.dataset = match.group(1)
            self.auroc = float(match.group(2))
            self.auroc_std = float(match.group(3))
            self.auprc = float(match.group(4))
            self.auprc_std = float(match.group(5))
            # FPR95 可能缺失
            if match.group(6) is not None:
                self.fpr95 = float(match.group(6))
                self.fpr95_std = float(match.group(7))
        else:
            # 如果没有最终结果统计，从每个试验中计算
            self._extract_from_trials()
        
        # 提取模型和数据集名称（如果还未提取）
        self._extract_model_dataset()
        
        # 状态：如果有 AUROC 或 AUPRC，则认为成功
        self.status = "OK" if (self.auroc > 0 or self.auprc > 0) else "FAILED"
        
        return {
            "model": self.model,
            "dataset": self.dataset,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "fpr95": self.fpr95,
            "auroc_std": self.auroc_std,
            "auprc_std": self.auprc_std,
            "fpr95_std": self.fpr95_std,
            "status": self.status,
            "trial_results": self.trial_results
        }
    
    def _extract_from_trials(self):
        """从每个试验的结果中提取指标并计算平均值和标准差"""
        # 找到最终结果统计的位置，只提取之前的内容
        final_stats_idx = self.log_content.find("最终结果统计")
        content_to_search = self.log_content[:final_stats_idx] if final_stats_idx != -1 else self.log_content
        
        auroc_values = []
        auprc_values = []
        fpr95_values = []
        
        # 提取所有试验的 AUROC（只提取0-1之间的小数，排除百分比格式）
        for m in self.TRIAL_AUROC_PATTERN.finditer(content_to_search):
            value_str = m.group(1)
            try:
                value = float(value_str)
                # 如果是0-1之间的小数，转换为百分比
                if 0 <= value <= 1:
                    auroc_values.append(value * 100)
                elif value > 1 and value <= 100:
                    # 已经是百分比格式
                    auroc_values.append(value)
            except ValueError:
                continue
        
        # 提取所有试验的 AUPRC
        for m in self.TRIAL_AUPRC_PATTERN.finditer(content_to_search):
            value_str = m.group(1)
            try:
                value = float(value_str)
                if 0 <= value <= 1:
                    auprc_values.append(value * 100)
                elif value > 1 and value <= 100:
                    auprc_values.append(value)
            except ValueError:
                continue
        
        # 提取所有试验的 FPR95
        for m in self.TRIAL_FPR95_PATTERN.finditer(content_to_search):
            value_str = m.group(1)
            try:
                value = float(value_str)
                if 0 <= value <= 1:
                    fpr95_values.append(value * 100)
                elif value > 1 and value <= 100:
                    fpr95_values.append(value)
            except ValueError:
                continue
        
        # 计算平均值和标准差
        if auroc_values:
            self.auroc = sum(auroc_values) / len(auroc_values)
            if len(auroc_values) > 1:
                variance = sum((x - self.auroc) ** 2 for x in auroc_values) / len(auroc_values)
                self.auroc_std = variance ** 0.5
        
        if auprc_values:
            self.auprc = sum(auprc_values) / len(auprc_values)
            if len(auprc_values) > 1:
                variance = sum((x - self.auprc) ** 2 for x in auprc_values) / len(auprc_values)
                self.auprc_std = variance ** 0.5
        
        if fpr95_values:
            self.fpr95 = sum(fpr95_values) / len(fpr95_values)
            if len(fpr95_values) > 1:
                variance = sum((x - self.fpr95) ** 2 for x in fpr95_values) / len(fpr95_values)
                self.fpr95_std = variance ** 0.5
        
        # 保存每个试验的结果
        num_trials = max(len(auroc_values), len(auprc_values), len(fpr95_values))
        for i in range(num_trials):
            trial_result = {}
            if i < len(auroc_values):
                trial_result['auroc'] = auroc_values[i]
            if i < len(auprc_values):
                trial_result['auprc'] = auprc_values[i]
            if i < len(fpr95_values):
                trial_result['fpr95'] = fpr95_values[i]
            if trial_result:
                self.trial_results.append(trial_result)
    
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
        if self.auroc > 0:
            metrics['AUROC'] = self.auroc
        if self.auprc > 0:
            metrics['AUPRC'] = self.auprc
        if self.fpr95 > 0:
            metrics['FPR95'] = self.fpr95
        return metrics
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        获取最好的指标数据（输出格式：指标：数据值）
        
        返回:
            包含最好结果的字典，格式为 {指标名: 数据值（平均值）}
        """
        best_metrics = {}
        if self.auroc > 0:
            best_metrics['AUROC'] = self.auroc
        if self.auprc > 0:
            best_metrics['AUPRC'] = self.auprc
        if self.fpr95 > 0:
            best_metrics['FPR95'] = self.fpr95
        return best_metrics
    
    @classmethod
    def from_file(cls, log_path: Path, model: str = None, dataset: str = None) -> "GOLDCollector":
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
    def from_string(cls, log_content: str, model: str = None, dataset: str = None) -> "GOLDCollector":
        """从字符串读取"""
        collector = cls(log_content)
        collector.collect()
        if model:
            collector.model = model
        if dataset:
            collector.dataset = dataset
        return collector

