import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List
import pandas as pd

class ImageMetricsVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn')
        
    def plot_roc_curves(self, metrics_results: Dict[str, Dict], title: str):
        """Plot ROC curves for all detectors"""
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
        """Plot Precision-Recall curves for all detectors"""
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
        """Plot heatmap comparing metrics across detectors"""
        metrics = [
            "accuracy", "precision", "recall", "f1",
            "false_accept_rate", "false_reject_rate",
            "equal_error_rate"
        ]
        detectors = list(metrics_results.keys())
        
        data = np.zeros((len(detectors), len(metrics)))
        for i, detector in enumerate(detectors):
            for j, metric in enumerate(metrics):
                data[i, j] = metrics_results[detector]["basic_metrics"][metric]
        
        plt.figure(figsize=(12, len(detectors) * 0.8 + 2))
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            xticklabels=metrics,
            yticklabels=detectors,
            cmap='YlOrRd'
        )
        plt.title(title)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_heatmap.png")
        plt.close()
        
    def plot_category_performance(
        self,
        metrics_results: Dict[str, Dict],
        category_distribution: Dict[str, Dict[str, int]],
        title: str
    ):
        """Plot performance metrics by category"""
        categories = list(category_distribution.keys())
        metrics = ["accuracy", "precision", "recall", "f1"]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title)
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            detector_names = list(metrics_results.keys())
            x = np.arange(len(categories))
            width = 0.8 / len(detector_names)
            
            for i, detector in enumerate(detector_names):
                values = []
                for category in categories:
                    cat_metrics = metrics_results[detector].get("category_metrics", {})
                    if category in cat_metrics:
                        values.append(
                            cat_metrics[category]["basic_metrics"][metric]
                        )
                    else:
                        values.append(0)
                
                ax.bar(
                    x + i * width - width * len(detector_names) / 2,
                    values,
                    width,
                    label=detector
                )
            
            ax.set_title(f"{metric.capitalize()} by Category")
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45)
            ax.set_ylim(0, 1)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_categories.png")
        plt.close()
        
    def plot_error_analysis(self, metrics_results: Dict[str, Dict], title: str):
        """Plot error analysis visualization"""
        detectors = list(metrics_results.keys())
        
        # Prepare data for error types
        error_types = ['false_positives', 'false_negatives']
        error_data = {
            detector: [
                results['confusion_matrix']['fp'],
                results['confusion_matrix']['fn']
            ] for detector, results in metrics_results.items()
        }
        
        # Create error type distribution plot
        plt.figure(figsize=(10, 6))
        x = np.arange(len(detectors))
        width = 0.35
        
        plt.bar(
            x - width/2,
            [error_data[d][0] for d in detectors],
            width,
            label='False Positives'
        )
        plt.bar(
            x + width/2,
            [error_data[d][1] for d in detectors],
            width,
            label='False Negatives'
        )
        
        plt.xlabel('Detectors')
        plt.ylabel('Number of Errors')
        plt.title(f'{title} - Error Analysis')
        plt.xticks(x, detectors, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_errors.png")
        plt.close()