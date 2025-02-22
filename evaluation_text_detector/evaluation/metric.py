from typing import List, Dict, Union
import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, average_precision_score,
    confusion_matrix
)

class AdvancedMetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred_proba: List[float]) -> Dict:
        """计算所有评估指标"""
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # 计算ROC曲线
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # 计算PR曲线
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # 计算不同阈值下的F1
        f1_scores = []
        thresholds = np.arange(0, 1.1, 0.1)
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision_t * recall_t) / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
            f1_scores.append(f1)
        
        # 找到最佳F1
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx]
        best_f1_score = f1_scores[best_f1_idx]
        
        # 计算默认阈值(0.5)下的指标
        y_pred = (y_pred_proba >= 0.5).astype(int)
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
                },
                "f1": {
                    "thresholds": thresholds.tolist(),
                    "scores": f1_scores,
                    "best": {
                        "threshold": float(best_f1_threshold),
                        "score": float(best_f1_score)
                    }
                }
            }
        }