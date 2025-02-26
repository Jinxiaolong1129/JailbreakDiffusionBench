# evaluation_text_detector/evaluation/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List
import pandas as pd

class MetricsVisualizer:
    def __init__(self, output_dir: str):
        """Initialize with output directory for saving visualizations
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
    def plot_roc_curves(self, metrics_results: Dict[str, Dict], title: str):
        """Plot ROC curves for multiple detectors
        
        Args:
            metrics_results: Dictionary mapping detector names to their metrics
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))
        
        for detector_name, results in metrics_results.items():
            if "error" in results:
                continue
                
            if "curves" in results and "roc" in results["curves"]:
                roc_data = results["curves"]["roc"]
                plt.plot(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    label=f'{detector_name} (AUC = {roc_data["auc"]:.3f})'
                )
        
        # Add random baseline
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save figure
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_roc.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_pr_curves(self, metrics_results: Dict[str, Dict], title: str):
        """Plot Precision-Recall curves for multiple detectors
        
        Args:
            metrics_results: Dictionary mapping detector names to their metrics
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))
        
        for detector_name, results in metrics_results.items():
            if "error" in results:
                continue
                
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
        plt.grid(True)
        
        # Save figure
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_pr.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics_heatmap(self, metrics_results: Dict[str, Dict], title: str):
        """Plot heatmap comparing different metrics across detectors
        
        Args:
            metrics_results: Dictionary mapping detector names to their metrics
            title: Title for the plot
        """
        # Define metrics to display
        metrics = ["accuracy", "precision", "recall", "f1", "auc", "average_precision"]
        
        # Filter out detectors with errors
        valid_detectors = {name: results for name, results in metrics_results.items() 
                         if "error" not in results}
        
        if not valid_detectors:
            print(f"No valid detector results to create heatmap for {title}")
            return
            
        detectors = list(valid_detectors.keys())
        
        # Prepare data for heatmap
        data = np.zeros((len(detectors), len(metrics)))
        for i, detector in enumerate(detectors):
            for j, metric in enumerate(metrics):
                if metric == "auc":
                    data[i, j] = valid_detectors[detector]["curves"]["roc"]["auc"] 
                elif metric == "average_precision":
                    data[i, j] = valid_detectors[detector]["curves"]["pr"]["average_precision"]
                elif metric == "f1":
                    data[i, j] = valid_detectors[detector]["curves"]["f1"]["best"]["score"]
                else:
                    data[i, j] = valid_detectors[detector]["basic_metrics"][metric]
        
        # Create heatmap
        plt.figure(figsize=(12, len(detectors) * 0.8 + 2))
        sns.heatmap(data, annot=True, fmt='.3f', 
                    xticklabels=metrics, 
                    yticklabels=detectors,
                    cmap='YlOrRd')
        plt.title(title)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrices(self, metrics_results: Dict[str, Dict], title: str):
        """Plot confusion matrices for all detectors
        
        Args:
            metrics_results: Dictionary mapping detector names to their metrics
            title: Title for the plot
        """
        # Filter out detectors with errors
        valid_detectors = {name: results for name, results in metrics_results.items() 
                         if "error" not in results}
        
        if not valid_detectors:
            print(f"No valid detector results for confusion matrices in {title}")
            return
            
        # Create subplots based on number of detectors
        n_detectors = len(valid_detectors)
        fig, axes = plt.subplots(1, n_detectors, figsize=(5*n_detectors, 5))
        
        # Handle case with only one detector
        if n_detectors == 1:
            axes = [axes]
            
        for ax, (detector_name, results) in zip(axes, valid_detectors.items()):
            cm = results["confusion_matrix"]
            cm_data = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            
            sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax,
                       xticklabels=["Benign", "Harmful"],
                       yticklabels=["Benign", "Harmful"])
            ax.set_title(f"{detector_name}")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
        
        plt.suptitle(f"{title} - Confusion Matrices")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_f1_threshold_curves(self, metrics_results: Dict[str, Dict], title: str):
        """Plot F1 scores across different thresholds
        
        Args:
            metrics_results: Dictionary mapping detector names to their metrics
            title: Title for the plot
        """
        plt.figure(figsize=(10, 8))
        
        for detector_name, results in metrics_results.items():
            if "error" in results:
                continue
                
            if "curves" in results and "f1" in results["curves"]:
                f1_data = results["curves"]["f1"]
                best_threshold = f1_data["best"]["threshold"]
                best_score = f1_data["best"]["score"]
                
                plt.plot(
                    f1_data["thresholds"],
                    f1_data["scores"],
                    label=f'{detector_name} (Best F1 = {best_score:.3f} at {best_threshold:.2f})'
                )
                plt.scatter([best_threshold], [best_score], marker='o', s=100, 
                           edgecolors='black', c='red')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(f"{title} - F1 Score vs Threshold")
        plt.legend(loc="best")
        plt.grid(True)
        
        # Save figure
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_f1_threshold.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_category_performance(self, category_metrics: Dict, title: str):
        """Plot performance metrics by category
        
        Args:
            category_metrics: Dictionary with category-specific metrics
            title: Title for the plot
        """
        if not category_metrics or "categories" not in category_metrics or not category_metrics["categories"]:
            print(f"No category data available for {title}")
            return
            
        categories = category_metrics["categories"]
        metrics_by_category = category_metrics["metrics"]
        
        # Extract metrics we want to plot
        metric_names = ["precision", "recall", "f1", "accuracy"]
        
        # Create DataFrame for easier plotting
        data = []
        for category in categories:
            if category in metrics_by_category:
                cat_data = metrics_by_category[category]
                for metric in metric_names:
                    if metric in cat_data["metrics"]:
                        data.append({
                            "Category": category,
                            "Metric": metric.capitalize(),
                            "Value": cat_data["metrics"][metric],
                            "Count": cat_data["count"]
                        })
        
        if not data:
            print(f"No metric data available for categories in {title}")
            return
            
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(14, 8))
        g = sns.barplot(x="Category", y="Value", hue="Metric", data=df)
        
        # Add count annotations
        for i, category in enumerate(sorted(df["Category"].unique())):
            count = df[df["Category"] == category]["Count"].iloc[0]
            plt.text(i, 0.05, f"n={count}", ha='center', fontweight='bold')
            
        plt.title(f"{title} - Performance by Category")
        plt.xlabel("Category")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Metric")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / f"{title.lower().replace(' ', '_')}_category_performance.png", dpi=300, bbox_inches='tight')
        plt.close()