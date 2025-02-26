import yaml
import json
from pathlib import Path
from typing import Dict, List, Union
import argparse
import datetime
from jailbreak_diffusion.judger.pre_checker.openai_text_moderation import OpenAITextDetector
from evaluation.data_loader import DatasetLoader
from evaluation.metric import TextMetricsCalculator
from evaluation.visualization import MetricsVisualizer

class TextBenchmarkRunner:
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
        """Create necessary directories with timestamp"""
        # Create timestamped experiment directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(self.config_path).stem
        self.output_dir = Path(self.config["output_dir"]) / f"{config_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a copy of the config file used
        with open(self.output_dir / "config_used.yaml", "w") as f:
            yaml.dump(self.config, f)
        
    def initialize_components(self):
        """Initialize all components"""
        self.detectors = self._init_detectors()
        self.data_loader = DatasetLoader(self.config["data_dir"])
        self.metrics_calculator = TextMetricsCalculator()
        self.visualizer = MetricsVisualizer(str(self.output_dir))
        
    def _init_detectors(self) -> Dict:
        """Initialize all detectors"""
        detectors = {}
        detector_configs = self.config["detectors"]
        
        # Initialize detector mapping - can be extended for more detectors
        detector_mapping = {
            "openai": OpenAITextDetector
        }
        
        for name, config in detector_configs.items():
            if name in detector_mapping:
                try:
                    detectors[name] = detector_mapping[name](**config)
                except Exception as e:
                    print(f"Error initializing {name} detector: {e}")
                    
        return detectors
        
    def evaluate_dataset(self, dataset_name: str, dataset_config: Dict) -> Dict:
        """Evaluate a single dataset with all detectors"""
        print(f"\nEvaluating dataset: {dataset_name}")
        
        # Create dataset specific output directory
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(exist_ok=True)
        self.visualizer.output_dir = Path(dataset_output_dir)
        
        # Load dataset
        dataset = self.data_loader.load_dataset(dataset_config["path"])
        dataset_info = self.data_loader.get_dataset_info(dataset_config["path"])
        print(f"Dataset info: {dataset_info}")
        
        # Save dataset info and original dataset JSON content
        with open(dataset_output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
            
        # Copy the original dataset JSON file for reference
        with open(dataset_config["path"], "r") as src_f:
            original_dataset = json.load(src_f)
            with open(dataset_output_dir / "original_dataset.json", "w") as dst_f:
                json.dump(original_dataset, dst_f, indent=2)
        
        results = {}
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name} on {dataset_name}...")
            
            try:
                # Get detection results
                texts = [prompt.text for prompt in dataset]
                raw_predictions = detector.check(texts)
                
                # Convert to binary predictions if needed
                if isinstance(raw_predictions[0], bool):
                    predictions = [1 if pred else 0 for pred in raw_predictions]
                else:
                    predictions = raw_predictions
                
                # Prepare evaluation data
                true_labels = [1 if p.label == "harmful" else 0 for p in dataset]
                categories = [p.category for p in dataset]
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(true_labels, predictions)
                results[detector_name] = metrics
                
                # Save detailed prediction results for each prompt
                detailed_predictions = []
                for i, (prompt, pred, true_label) in enumerate(zip(dataset, predictions, true_labels)):
                    detailed_predictions.append({
                        "id": prompt.id,
                        "text": prompt.text,
                        "actual_label": "harmful" if true_label == 1 else "benign",
                        "predicted_label": "harmful" if int(pred) == 1 else "benign",
                        "correct": int(pred) == true_label,
                        "category": prompt.category,
                        "source": prompt.source
                    })
                
                # Save detailed results
                detailed_results = {
                    "detector": detector_name,
                    "dataset": dataset_name,
                    "total_prompts": len(detailed_predictions),
                    "correctly_classified": sum(1 for p in detailed_predictions if p["correct"]),
                    "metrics": metrics["basic_metrics"],
                    "predictions": detailed_predictions
                }
                
                # Save detector results to its own file
                detector_output_file = dataset_output_dir / f"{detector_name}_results.json"
                detailed_output_file = dataset_output_dir / f"{detector_name}_detailed_predictions.json"
                
                with open(detector_output_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
                with open(detailed_output_file, "w") as f:
                    json.dump(detailed_results, f, indent=2)
                
                
            except Exception as e:
                print(f"Error evaluating {detector_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[detector_name] = {"error": str(e)}
                
        # Generate visualizations for this dataset
        try:
            self.visualizer.plot_roc_curves(results, f"{dataset_name} ROC Curves")
            self.visualizer.plot_pr_curves(results, f"{dataset_name} PR Curves")
            self.visualizer.plot_metrics_heatmap(results, f"{dataset_name} Metrics Comparison")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
                
        return results
    
    
    def run_evaluation(self):
        """Run complete evaluation pipeline for all datasets and detectors"""
        all_results = {}
        
        # Evaluate each dataset
        for dataset_name, dataset_config in self.config["datasets"].items():
            dataset_results = self.evaluate_dataset(dataset_name, dataset_config)
            all_results[dataset_name] = dataset_results
        
        # Create overall summary
        self._create_experiment_summary(all_results)
        
        return all_results
    
    def _create_experiment_summary(self, all_results: Dict):
        """Create a summary of the experiment across all datasets and detectors"""
        summary = {
            "config_file": self.config_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "datasets": list(self.config["datasets"].keys()),
            "detectors": list(self.config["detectors"].keys()),
            "results_summary": {}
        }
        
        # Create a summary table of key metrics for all detectors across all datasets
        for dataset_name, dataset_results in all_results.items():
            summary["results_summary"][dataset_name] = {}
            
            for detector_name, metrics in dataset_results.items():
                if "error" in metrics:
                    summary["results_summary"][dataset_name][detector_name] = {
                        "error": metrics["error"]
                    }
                    continue
                    
                # Extract key metrics
                summary["results_summary"][dataset_name][detector_name] = {
                    "accuracy": metrics["basic_metrics"]["accuracy"],
                    "precision": metrics["basic_metrics"]["precision"],
                    "recall": metrics["basic_metrics"]["recall"],
                    "f1": metrics["basic_metrics"].get("f1", 0.0),
                    "roc_auc": metrics["curves"]["roc"]["auc"],
                    "average_precision": metrics["curves"]["pr"]["average_precision"]
                }
        
        # Save summary to file
        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate summary tables as text
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            f.write(f"Experiment Summary\n")
            f.write(f"=================\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Date: {summary['timestamp']}\n\n")
            
            for dataset_name in summary["datasets"]:
                f.write(f"\nDataset: {dataset_name}\n")
                f.write("-" * (len(dataset_name) + 10) + "\n")
                
                # Create header
                metrics_header = ["Detector", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"]
                f.write(" | ".join(metrics_header) + "\n")
                f.write("-" * 80 + "\n")
                
                # Add data rows
                for detector_name in summary["detectors"]:
                    if detector_name in summary["results_summary"][dataset_name]:
                        detector_metrics = summary["results_summary"][dataset_name][detector_name]
                        
                        if "error" in detector_metrics:
                            row = [detector_name, "ERROR", "", "", "", "", ""]
                        else:
                            row = [
                                detector_name,
                                f"{detector_metrics['accuracy']:.4f}",
                                f"{detector_metrics['precision']:.4f}",
                                f"{detector_metrics['recall']:.4f}",
                                f"{detector_metrics['f1']:.4f}",
                                f"{detector_metrics['roc_auc']:.4f}",
                                f"{detector_metrics['average_precision']:.4f}"
                            ]
                        
                        f.write(" | ".join(row) + "\n")
                
                f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Run text detector evaluation")
    parser.add_argument(
        "--config", 
        default="evaluation_text_detector/config/text_evaluation_configs.yaml",
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run evaluation on all config files in the config directory"
    )
    args = parser.parse_args()
    
    if args.all_configs:
        # Run all config files in the config directory
        config_dir = Path("evaluation_text_detector/config")
        config_files = list(config_dir.glob("*.yaml"))
        
        for config_file in config_files:
            print(f"\n\nRunning evaluation with config: {config_file}")
            runner = TextBenchmarkRunner(str(config_file))
            runner.run_evaluation()
    else:
        # Run evaluation with specified config
        runner = TextBenchmarkRunner(args.config)
        all_results = runner.run_evaluation()
        
        # Print high-level results summary
        print("\nEvaluation Results Summary:")
        for dataset_name, results in all_results.items():
            print(f"\nDataset: {dataset_name}")
            for detector_name, metrics in results.items():
                if "error" in metrics:
                    print(f"{detector_name}: Error - {metrics['error']}")
                    continue
                    
                print(f"\n{detector_name}:")
                print("Basic Metrics:")
                for metric_name, value in metrics["basic_metrics"].items():
                    print(f"  {metric_name}: {value:.4f}")
                print(f"ROC AUC: {metrics['curves']['roc']['auc']:.4f}")
                print(f"Average Precision: {metrics['curves']['pr']['average_precision']:.4f}")

if __name__ == "__main__":
    main()