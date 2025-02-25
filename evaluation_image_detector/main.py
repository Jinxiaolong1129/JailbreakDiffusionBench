import yaml
import json
from pathlib import Path
from typing import Dict, List, Union
import argparse
from PIL import Image
from jailbreak_diffusion.judger.post_checker import (
    OpenAIImageDetector,
)
from evaluation.data_loader import ImageDatasetLoader
from evaluation.metric import ImageMetricsCalculator
from evaluation.visualization import ImageMetricsVisualizer

class ImageBenchmarkRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        self.initialize_components()
        
    def load_config(self):
        """Load configuration file"""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_components(self):
        """Initialize all components"""
        self.detectors = self._init_detectors()
        self.data_loader = ImageDatasetLoader(self.config["data_dir"])
        self.metrics_calculator = ImageMetricsCalculator()
        self.visualizer = ImageMetricsVisualizer(self.output_dir)
        
    def _init_detectors(self) -> Dict:
        """Initialize all detectors"""
        detectors = {}
        detector_configs = self.config["detectors"]
        
        detector_mapping = {
            "openai": OpenAIImageDetector
        }
        
        for name, config in detector_configs.items():
            if name in detector_mapping:
                try:
                    detectors[name] = detector_mapping[name](**config)
                except Exception as e:
                    print(f"Error initializing {name} detector: {e}")
                    
        return detectors
        
    def evaluate_dataset(self, dataset_name: str, dataset_config: Dict) -> Dict:
        """Evaluate a single dataset"""
        print(f"\nEvaluating dataset: {dataset_name}")
        
        # Load dataset
        dataset = self.data_loader.load_dataset(dataset_config["path"])
        dataset_info = self.data_loader.get_dataset_info(dataset_config["path"])
        print(f"Dataset info: {dataset_info}")
        
        results = {}
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name}...")
            
            try:
                # Get detection results
                images = [item.image for item in dataset]
                predictions = detector.check(images)
                
                # Prepare evaluation data
                true_labels = [1 if item.label == "unsafe" else 0 for item in dataset]
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(true_labels, predictions)
                results[detector_name] = metrics
                
                # Save misclassified examples if configured
                if self.config.get("save_misclassified", False):
                    self._save_misclassified_examples(
                        dataset, predictions, true_labels,
                        detector_name, dataset_name
                    )
                
            except Exception as e:
                print(f"Error evaluating {detector_name}: {e}")
                results[detector_name] = {"error": str(e)}
                
        return results
    
    def _save_misclassified_examples(
        self, dataset: List, predictions: List[bool],
        true_labels: List[int], detector_name: str,
        dataset_name: str
    ):
        """Save misclassified images for analysis"""
        misclassified_dir = self.output_dir / "misclassified" / dataset_name / detector_name
        misclassified_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (pred, true, item) in enumerate(zip(predictions, true_labels, dataset)):
            if pred != true:
                error_type = "false_positive" if pred == 1 else "false_negative"
                save_path = misclassified_dir / error_type / f"{item.id}.png"
                save_path.parent.mkdir(exist_ok=True)
                item.image.save(save_path)
                
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        all_results = {}
        
        # Evaluate each dataset
        for dataset_name, dataset_config in self.config["datasets"].items():
            results = self.evaluate_dataset(dataset_name, dataset_config)
            all_results[dataset_name] = results
            
            # Save results
            output_file = self.output_dir / f"{dataset_name}_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Generate visualizations for this dataset
            self.visualizer.plot_roc_curves(results, f"{dataset_name} ROC Curves")
            self.visualizer.plot_pr_curves(results, f"{dataset_name} PR Curves")
            self.visualizer.plot_metrics_heatmap(results, f"{dataset_name} Metrics Comparison")
            

        # Calculate weighted average results
        overall_results = self.calculate_overall_results(all_results)
        
        # Save overall results
        with open(self.output_dir / "overall_results.json", "w") as f:
            json.dump(overall_results, f, indent=2)
            
        # Generate overall visualizations
        self.visualizer.plot_metrics_heatmap(
            overall_results, 
            "Overall Metrics Comparison"
        )
        
        return all_results, overall_results
    
    def calculate_overall_results(self, all_results: Dict) -> Dict:
        """Calculate weighted average results across all datasets"""
        overall_results = {}
        
        for detector_name in self.detectors.keys():
            detector_metrics = {
                "basic_metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "false_positive_rate": 0.0,
                    "false_negative_rate": 0.0,
                    "false_accept_rate": 0.0,  
                    "false_reject_rate": 0.0,  
                    "equal_error_rate": 0.0   
                },
                "curves": {
                    "roc": {"auc": 0.0},
                    "pr": {"average_precision": 0.0}
                }
            }
            
            total_weight = 0
            for dataset_name, results in all_results.items():
                if detector_name not in results or "error" in results[detector_name]:
                    continue
                    
                weight = self.config["datasets"][dataset_name]["weight"]
                total_weight += weight
                
                # Accumulate basic metrics
                for metric in detector_metrics["basic_metrics"]:
                    detector_metrics["basic_metrics"][metric] += (
                        results[detector_name]["basic_metrics"][metric] * weight
                    )
                
                # Accumulate curve metrics
                detector_metrics["curves"]["roc"]["auc"] += (
                    results[detector_name]["curves"]["roc"]["auc"] * weight
                )
                detector_metrics["curves"]["pr"]["average_precision"] += (
                    results[detector_name]["curves"]["pr"]["average_precision"] * weight
                )
            
            # Calculate weighted averages
            if total_weight > 0:
                for metric in detector_metrics["basic_metrics"]:
                    detector_metrics["basic_metrics"][metric] /= total_weight
                    
                detector_metrics["curves"]["roc"]["auc"] /= total_weight
                detector_metrics["curves"]["pr"]["average_precision"] /= total_weight
                
            overall_results[detector_name] = detector_metrics
            
        return overall_results

def main():
    parser = argparse.ArgumentParser(description="Run image detector evaluation")
    parser.add_argument(
        "--config", 
        default="config/image_evaluation_configs.yaml",
        help="Path to evaluation config file"
    )
    args = parser.parse_args()
    
    # Run evaluation
    runner = ImageBenchmarkRunner(args.config)
    all_results, overall_results = runner.run_evaluation()
    
    # Print overall results
    print("\nOverall Results:")
    for detector_name, metrics in overall_results.items():
        print(f"\n{detector_name}:")
        print("Basic Metrics:")
        for metric, value in metrics["basic_metrics"].items():
            print(f"  {metric}: {value:.3f}")
        print("ROC AUC:", metrics["curves"]["roc"]["auc"])
        print("Average Precision:", metrics["curves"]["pr"]["average_precision"])

if __name__ == "__main__":
    main()