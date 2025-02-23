import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict

class ImageMetricsVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn')
        
    def plot_roc_curves(self, metrics_results: Dict[str, Dict], title: str):
        plt.figure(figsize=(10, 8))
        
        for detector_name, results in metrics_results.items():
            if "curves" in results and "roc" in results["curves"]:
                roc_data = results["curves"]["roc"]
                plt.plot(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    label=f'{detector_name} (AUC = {roc_data["auc"]:.3f})'
                )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_roc.png")
        plt.close()
        
    def plot_pr_curves(self, metrics_results: Dict[str, Dict], title: str):
        plt.figure(figsize=(10, 8))
        
        for detector_name, results in metrics_results.items():
            if "curves" in results and "pr" in results["curves"]:
                pr_data = results["curves"]["pr"]
                plt.plot(
                    pr_data["recall"],
                    pr_data["precision"],
                    label=f'{detector_name} (AP = {pr_data["average_precision"]:.3f})'
                )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_pr.png")
        plt.close()
        
    def plot_metrics_heatmap(self, metrics_results: Dict[str, Dict], title: str):
        """绘制不同检测器的指标热力图"""
        metrics = ["accuracy", "precision", "recall", "f1", "auc", "average_precision"]
        detectors = list(metrics_results.keys())
        
        data = np.zeros((len(detectors), len(metrics)))
        for i, detector in enumerate(detectors):
            for j, metric in enumerate(metrics):
                if metric in ["auc", "average_precision"]:
                    data[i, j] = metrics_results[detector]["curves"]["roc"]["auc"] if metric == "auc" else \
                                 metrics_results[detector]["curves"]["pr"]["average_precision"]
                else:
                    data[i, j] = metrics_results[detector]["basic_metrics"][metric]
        
        plt.figure(figsize=(12, len(detectors) * 0.8 + 2))
        sns.heatmap(data, annot=True, fmt='.3f', 
                    xticklabels=metrics, 
                    yticklabels=detectors,
                    cmap='YlOrRd')
        plt.title(title)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_heatmap.png")
        plt.close()

    def plot_confusion_matrices(self, metrics_results: Dict[str, Dict], title: str):
        """绘制所有检测器的混淆矩阵"""
        num_detectors = len(metrics_results)
        fig, axes = plt.subplots(1, num_detectors, figsize=(5*num_detectors, 5))
        
        for idx, (detector_name, results) in enumerate(metrics_results.items()):
            cm = results["confusion_matrix"]
            cm_data = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            
            ax = axes[idx] if num_detectors > 1 else axes
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{detector_name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticklabels(["Safe", "Unsafe"])
            ax.set_yticklabels(["Safe", "Unsafe"])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_confusion_matrices.png")
        plt.close()