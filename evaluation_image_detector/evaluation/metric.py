from typing import List, Dict
import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, average_precision_score,
    confusion_matrix
)

class ImageMetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[bool]) -> Dict:
        """计算所有评估指标"""
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).astype(int)
        
        # 计算ROC曲线
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # 计算PR曲线
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_pred)
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "basic_metrics": {
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0
            },
            "curves": {
                "roc": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                    "auc": float(roc_auc)
                },
                "pr": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                    "average_precision": float(avg_precision)
                }
            }
        }