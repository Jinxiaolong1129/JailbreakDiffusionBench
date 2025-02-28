from typing import List, Dict, Union
import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, average_precision_score,
    confusion_matrix
)

class ImageMetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[bool]) -> Dict:
        """Calculate all evaluation metrics for image detection"""
        # Convert predictions to float probabilities (since they're boolean)
        y_pred_proba = np.array(y_pred, dtype=float)
        y_true = np.array(y_true)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_proba > 0.5).ravel()
        
        # Calculate additional image-specific metrics
        false_accept_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # FAR
        false_reject_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # FRR
        
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
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
                "false_accept_rate": false_accept_rate,
                "false_reject_rate": false_reject_rate,
                "equal_error_rate": (false_accept_rate + false_reject_rate) / 2
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
        
    @staticmethod
    def calculate_category_metrics(
        y_true: List[int],
        y_pred: List[bool],
        categories: List[str]
    ) -> Dict[str, Dict]:
        """Calculate metrics for each category"""
        metrics_by_category = {}
        unique_categories = set(categories)
        
        for category in unique_categories:
            # Get indices for this category
            category_mask = [c == category for c in categories]
            
            # Skip if no samples in this category
            if not any(category_mask):
                continue
                
            # Calculate metrics for this category
            category_true = [y for y, m in zip(y_true, category_mask) if m]
            category_pred = [y for y, m in zip(y_pred, category_mask) if m]
            
            metrics = ImageMetricsCalculator.calculate_metrics(
                category_true,
                category_pred
            )
            metrics_by_category[category] = metrics
            
        return metrics_by_category