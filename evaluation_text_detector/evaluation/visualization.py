import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List

class MetricsVisualizer:
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