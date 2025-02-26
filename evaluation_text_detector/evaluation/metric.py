# evaluation_text_detector/evaluation/metric.py
from typing import List, Dict, Union
import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, average_precision_score,
    confusion_matrix, f1_score
)

class AdvancedMetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[float]) -> Dict:
        """Calculate comprehensive evaluation metrics
        
        Args:
            y_true: List of ground truth labels (1 for harmful, 0 for benign)
            y_pred: List of prediction scores or binary predictions
            
        Returns:
            Dict with all calculated metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Check if predictions are probabilistic or binary
        if np.all(np.logical_or(y_pred == 0, y_pred == 1)):
            # Binary predictions - convert to float array for consistency
            y_pred_binary = y_pred
            y_pred_proba = y_pred.astype(float)
        else:
            # Probabilistic predictions - convert to binary with 0.5 threshold
            y_pred_binary = (y_pred >= 0.5).astype(int)
            y_pred_proba = y_pred
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Calculate confusion matrix and basic metrics using the binary predictions
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Calculate additional F1 scores at different thresholds if predictions are probabilistic
        f1_scores = []
        if not np.all(np.logical_or(y_pred == 0, y_pred == 1)):
            thresholds = np.arange(0, 1.1, 0.1)
            for threshold in thresholds:
                y_pred_t = (y_pred_proba >= threshold).astype(int)
                f1_t = f1_score(y_true, y_pred_t, zero_division=0)
                f1_scores.append(f1_t)
                
            # Find the best F1 threshold
            best_f1_idx = np.argmax(f1_scores)
            best_f1_threshold = thresholds[best_f1_idx]
            best_f1_score = f1_scores[best_f1_idx]
        else:
            # If predictions are already binary, use fixed values
            thresholds = [0.5]
            f1_scores = [f1]
            best_f1_threshold = 0.5
            best_f1_score = f1
        
        return {
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "basic_metrics": {
                "accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "f1": f1,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
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
                    "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else [0.5],
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