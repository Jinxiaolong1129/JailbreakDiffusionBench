import yaml
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple
import argparse
from PIL import Image
import os
import datetime
from jailbreak_diffusion.judger.post_checker import (
    MultiheadDetector,
    Q16Detector,
    FinetunedQ16Detector,  
    SD_SafetyCheckerDetector,
    OpenAIImageDetector,
    AzureImageDetector,
    GoogleImageDetector, 
    AWSImageDetector,
    LlavaGuardChecker
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
        # Create timestamped experiment directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(self.config_path).stem
        self.output_dir = Path(self.config["output_dir"]) / f"{config_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category directories
        model = self.config["model_name"]
        model_dir = self.output_dir / model
        model_dir.mkdir(exist_ok=True)
        
        for category in self.config["datasets"].keys():
            category_dir = model_dir / category
            category_dir.mkdir(exist_ok=True)
        
        # Save a copy of the config file used
        with open(self.output_dir / "config_used.yaml", "w") as f:
            yaml.dump(self.config, f)
        
    def initialize_components(self):
        """Initialize all components"""
        self.detectors = self._init_detectors()
        self.data_loader = ImageDatasetLoader(self.config["data_dir"])
        self.metrics_calculator = ImageMetricsCalculator()
        self.visualizer = ImageMetricsVisualizer(str(self.output_dir))
        
    def _init_detectors(self) -> Dict:
        """Initialize all detectors"""
        detectors = {}
        detector_configs = self.config["detectors"]
        
        detector_mapping = {
            "openai_image_checker": OpenAIImageDetector,
            "q16_detector": Q16Detector,
            "multihead_detector": MultiheadDetector,
            "finetuned_q16_detector": FinetunedQ16Detector,
            "sd_safety_checker": SD_SafetyCheckerDetector,
            "azure_image_checker": AzureImageDetector,
            "google_image_checker": GoogleImageDetector,
            "aws_image_checker": AWSImageDetector,
            "llava_guard": LlavaGuardChecker
        }
        
        for name, config in detector_configs.items():
            if name in detector_mapping:
                try:
                    # Handle empty config
                    if not config:
                        config = {}
                        
                    detectors[name] = detector_mapping[name](**config)
                    print(f"Successfully initialized {name} detector")
                except Exception as e:
                    print(f"Error initializing {name} detector: {e}")
                    import traceback
                    traceback.print_exc()
                    
        return detectors
    
    def _get_flattened_datasets(self) -> List[Tuple[str, str, Dict]]:
        """
        Convert nested dataset structure to flattened list
        Returns: [(category, dataset_name, dataset_config), ...]
        """
        flattened_datasets = []
        model = self.config["model_name"]
        for category, datasets in self.config["datasets"].items():
            for dataset_name, dataset_config in datasets.items():
                flattened_datasets.append((category, dataset_name, dataset_config))
        return flattened_datasets
        
    def evaluate_dataset(self, category: str, dataset_name: str, dataset_config: Dict) -> Dict:
        """Evaluate a single dataset with all detectors"""
        model = self.config["model_name"]
        print(f"\nEvaluating dataset: {model}/{category}/{dataset_name}")
        
        # Create dataset specific output directory
        dataset_output_dir = self.output_dir / model / category / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualizer output directory
        self.visualizer.output_dir = dataset_output_dir
        
        # Load dataset
        dataset = self.data_loader.load_dataset(dataset_config["path"])
        dataset_info = self.data_loader.get_dataset_info(dataset_config["path"])
        print(f"Dataset info: {dataset_info}")
        
        # Save dataset info and original dataset JSON content
        with open(dataset_output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
            
        # Copy the original dataset JSON file for reference if it exists
        try:
            metadata_path = Path(dataset_config["path"]).parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as src_f:
                    metadata = json.load(src_f)
                    with open(dataset_output_dir / "original_metadata.json", "w") as dst_f:
                        json.dump(metadata, dst_f, indent=2)
        except Exception as e:
            print(f"Warning: Could not copy original metadata: {e}")
        
        results = {}
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name} on {model}/{category}/{dataset_name}...")
            
            try:
                # Get detection results
                images = [item.image for item in dataset]
                
                # Use batch processing if supported
                if hasattr(detector, 'batch_check') and callable(getattr(detector, 'batch_check')):
                    batch_size = self.config.get("batch_size", 16)
                    predictions = []
                    for i in range(0, len(images), batch_size):
                        batch_images = images[i:i+batch_size]
                        batch_preds = detector.batch_check(batch_images)
                        predictions.extend(batch_preds)
                        print(f"Processed batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}")
                else:
                    # Process individually
                    predictions = []
                    for idx, image in enumerate(images):
                        try:
                            pred = detector.check(image)
                            predictions.append(pred)
                            if (idx + 1) % 10 == 0:
                                print(f"Processed {idx + 1}/{len(images)} images")
                        except Exception as e:
                            print(f"Error processing image {idx} with {detector_name}: {e}")
                            # Use default value (conservative approach: assume safe)
                            predictions.append(0)
                
                # Ensure predictions length matches images length
                if len(predictions) != len(images):
                    print(f"Warning: predictions length ({len(predictions)}) doesn't match images length ({len(images)})")
                    # Fill missing predictions
                    while len(predictions) < len(images):
                        predictions.append(0)
                    # Truncate extra predictions
                    predictions = predictions[:len(images)]
                
                # Prepare evaluation data
                true_labels = [1 if item.label.lower() in ["unsafe", "harmful"] else 0 for item in dataset]
                categories = [item.category for item in dataset]
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(true_labels, predictions)
                results[detector_name] = metrics
                
                # Save detailed prediction results
                detailed_predictions = []
                for i, (item, pred, true_label) in enumerate(zip(dataset, predictions, true_labels)):
                    pred_binary = 1 if pred >= 0.5 else 0 if isinstance(pred, float) else int(pred)
                    
                    detailed_predictions.append({
                        "id": item.id,
                        "file_name": item.file_name,
                        "actual_label": "unsafe" if true_label == 1 else "safe",
                        "predicted_label": "unsafe" if pred_binary == 1 else "safe",
                        "prediction_score": float(pred) if isinstance(pred, (int, float)) else None,
                        "correct": pred_binary == true_label,
                        "category": item.category,
                        "source": item.source
                    })
                
                # Save detailed results
                detailed_results = {
                    "detector": detector_name,
                    "model": model,
                    "category": category,
                    "dataset": dataset_name,
                    "total_images": len(detailed_predictions),
                    "correctly_classified": sum(1 for p in detailed_predictions if p["correct"]),
                    "metrics": metrics["basic_metrics"],
                    "predictions": detailed_predictions
                }
                
                # Save detector results
                detector_output_file = dataset_output_dir / f"{detector_name}_results.json"
                detailed_output_file = dataset_output_dir / f"{detector_name}_detailed_predictions.json"
                
                with open(detector_output_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
                with open(detailed_output_file, "w") as f:
                    json.dump(detailed_results, f, indent=2)
                
                # Save misclassified examples if configured
                if self.config.get("save_misclassified", False):
                    self._save_misclassified_examples(
                        dataset, predictions, true_labels,
                        detector_name, dataset_output_dir
                    )
                
            except Exception as e:
                print(f"Error evaluating {detector_name} on {model}/{category}/{dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[detector_name] = {"error": str(e)}
                
        # Generate visualizations
        try:
            self.visualizer.plot_roc_curves(results, f"{model}/{category}/{dataset_name} ROC Curves")
            self.visualizer.plot_pr_curves(results, f"{model}/{category}/{dataset_name} PR Curves")
            if hasattr(self.visualizer, 'plot_metrics_heatmap'):
                self.visualizer.plot_metrics_heatmap(results, f"{model}/{category}/{dataset_name} Metrics Comparison")
            if hasattr(self.visualizer, 'plot_error_analysis'):
                self.visualizer.plot_error_analysis(results, f"{model}/{category}/{dataset_name} Error Analysis")
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
                
        return results
    
    def _save_misclassified_examples(
        self, dataset: List, predictions: List,
        true_labels: List[int], detector_name: str,
        output_dir: Path
    ):
        """Save misclassified images for analysis"""
        misclassified_dir = output_dir / "misclassified" / detector_name
        misclassified_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (pred, true, item) in enumerate(zip(predictions, true_labels, dataset)):
            pred_binary = 1 if pred >= 0.5 else 0 if isinstance(pred, float) else int(pred)
            if pred_binary != true:
                error_type = "false_positive" if pred_binary == 1 else "false_negative"
                save_dir = misclassified_dir / error_type
                save_dir.mkdir(exist_ok=True)
                
                try:
                    # Save original image
                    save_path = save_dir / f"{item.id}_{item.file_name}"
                    item.image.save(save_path)
                    
                    # Save metadata
                    with open(save_dir / f"{item.id}_metadata.json", "w") as f:
                        metadata = {
                            "id": item.id,
                            "file_name": item.file_name,
                            "actual_label": "unsafe" if true == 1 else "safe",
                            "predicted_label": "unsafe" if pred_binary == 1 else "safe",
                            "prediction_score": float(pred) if isinstance(pred, (int, float)) else None,
                            "category": item.category,
                            "source": item.source
                        }
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    print(f"Error saving misclassified example {item.id}: {e}")
                
    def run_evaluation(self):
        """Run complete evaluation pipeline for all datasets and detectors"""
        all_results = {}
        model = self.config["model_name"]
        
        # Get flattened dataset list
        flattened_datasets = self._get_flattened_datasets()
        
        # Organize results by category/dataset
        all_results[model] = {}
        for category, dataset_name, dataset_config in flattened_datasets:
            # Ensure category exists in results dict
            if category not in all_results[model]:
                all_results[model][category] = {}
                
            # Run evaluation
            dataset_results = self.evaluate_dataset(category, dataset_name, dataset_config)
            all_results[model][category][dataset_name] = dataset_results
        
        # Create overall summary
        self._create_experiment_summary(all_results)
        
        return all_results
    
    def _create_experiment_summary(self, all_results: Dict):
        """Create a summary of the experiment across all datasets and detectors"""
        model = self.config["model_name"]
        
        # Initialize summary
        summary = {
            "config_file": self.config_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model,
            "categories": list(self.config["datasets"].keys()),
            "detectors": list(self.config["detectors"].keys()),
            "results_summary": all_results[model]
        }
        
        # Calculate category averages
        category_averages = {}
        for category, datasets in all_results[model].items():
            category_averages[category] = {}
            
            for detector_name in summary["detectors"]:
                # Collect metrics for this detector across all datasets in the category
                detector_metrics = []
                
                for dataset_name, detector_results in datasets.items():
                    if detector_name in detector_results and "error" not in detector_results[detector_name]:
                        detector_metrics.append(detector_results[detector_name]["basic_metrics"])
                
                if detector_metrics:
                    # Calculate averages
                    category_averages[category][detector_name] = {
                        "accuracy": sum(m["accuracy"] for m in detector_metrics) / len(detector_metrics),
                        "precision": sum(m["precision"] for m in detector_metrics) / len(detector_metrics),
                        "recall": sum(m["recall"] for m in detector_metrics) / len(detector_metrics),
                        "f1": sum(m["f1"] for m in detector_metrics) / len(detector_metrics),
                        "roc_auc": sum(detector_results[detector_name]["curves"]["roc"]["auc"] for dataset_name, detector_results in datasets.items() 
                                 if detector_name in detector_results and "error" not in detector_results[detector_name]) / len(detector_metrics),
                        "average_precision": sum(detector_results[detector_name]["curves"]["pr"]["average_precision"] for dataset_name, detector_results in datasets.items() 
                                           if detector_name in detector_results and "error" not in detector_results[detector_name]) / len(detector_metrics)
                    }
        
        # Add category averages to summary
        summary["category_averages"] = category_averages
        
        # Calculate model-wide averages
        model_averages = {}
        for detector_name in summary["detectors"]:
            # Collect metrics for this detector across all categories
            all_detector_metrics = []
            
            for category in category_averages:
                if detector_name in category_averages[category]:
                    all_detector_metrics.append(category_averages[category][detector_name])
            
            if all_detector_metrics:
                # Calculate averages
                model_averages[detector_name] = {
                    "accuracy": sum(m["accuracy"] for m in all_detector_metrics) / len(all_detector_metrics),
                    "precision": sum(m["precision"] for m in all_detector_metrics) / len(all_detector_metrics),
                    "recall": sum(m["recall"] for m in all_detector_metrics) / len(all_detector_metrics),
                    "f1": sum(m["f1"] for m in all_detector_metrics) / len(all_detector_metrics),
                    "roc_auc": sum(m["roc_auc"] for m in all_detector_metrics) / len(all_detector_metrics),
                    "average_precision": sum(m["average_precision"] for m in all_detector_metrics) / len(all_detector_metrics)
                }
        
        # Add model averages to summary
        summary["model_averages"] = model_averages
        
        # Save summary to file
        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate text format summary tables
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            f.write(f"Experiment Summary\n")
            f.write(f"=================\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Date: {summary['timestamp']}\n")
            f.write(f"Model: {model}\n\n")
            
            # For each category
            for category in summary["categories"]:
                f.write(f"\nCategory: {category}\n")
                f.write("-" * (len(category) + 10) + "\n\n")
                
                # For each dataset in the category
                for dataset_name in summary["results_summary"][category].keys():
                    f.write(f"Dataset: {dataset_name}\n")
                    f.write("~" * (len(dataset_name) + 9) + "\n")
                    
                    # Create header
                    metrics_header = ["Detector", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"]
                    f.write(" | ".join(metrics_header) + "\n")
                    f.write("-" * 80 + "\n")
                    
                    # Add data rows
                    for detector_name in summary["detectors"]:
                        if detector_name in summary["results_summary"][category][dataset_name]:
                            detector_results = summary["results_summary"][category][dataset_name][detector_name]
                            
                            if "error" in detector_results:
                                row = [detector_name, "ERROR", "", "", "", "", ""]
                            else:
                                metrics = detector_results["basic_metrics"]
                                row = [
                                    detector_name,
                                    f"{metrics['accuracy']:.4f}",
                                    f"{metrics['precision']:.4f}",
                                    f"{metrics['recall']:.4f}",
                                    f"{metrics['f1']:.4f}",
                                    f"{detector_results['curves']['roc']['auc']:.4f}",
                                    f"{detector_results['curves']['pr']['average_precision']:.4f}"
                                ]
                            
                            f.write(" | ".join(row) + "\n")
                    
                    f.write("\n")
                
                # Print category average
                f.write(f"Category Average: {category}\n")
                f.write("~" * (len(category) + 17) + "\n")
                
                # Create header
                metrics_header = ["Detector", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"]
                f.write(" | ".join(metrics_header) + "\n")
                f.write("-" * 80 + "\n")
                
                # Add category average rows
                for detector_name in summary["detectors"]:
                    if detector_name in category_averages[category]:
                        avg_metrics = category_averages[category][detector_name]
                        row = [
                            detector_name,
                            f"{avg_metrics['accuracy']:.4f}",
                            f"{avg_metrics['precision']:.4f}",
                            f"{avg_metrics['recall']:.4f}",
                            f"{avg_metrics['f1']:.4f}",
                            f"{avg_metrics['roc_auc']:.4f}",
                            f"{avg_metrics['average_precision']:.4f}"
                        ]
                        f.write(" | ".join(row) + "\n")
                
                f.write("\n")
            
            # Print model average
            f.write(f"Model Average: {model}\n")
            f.write("=" * (len(model) + 15) + "\n")
            
            # Create header
            metrics_header = ["Detector", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"]
            f.write(" | ".join(metrics_header) + "\n")
            f.write("-" * 80 + "\n")
            
            # Add model average rows
            for detector_name in summary["detectors"]:
                if detector_name in model_averages:
                    avg_metrics = model_averages[detector_name]
                    row = [
                        detector_name,
                        f"{avg_metrics['accuracy']:.4f}",
                        f"{avg_metrics['precision']:.4f}",
                        f"{avg_metrics['recall']:.4f}",
                        f"{avg_metrics['f1']:.4f}",
                        f"{avg_metrics['roc_auc']:.4f}",
                        f"{avg_metrics['average_precision']:.4f}"
                    ]
                    f.write(" | ".join(row) + "\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run image detector evaluation")
    parser.add_argument(
        "--config", 
        default="evaluation_image_detector/config/openai_image_checker.yaml",
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run evaluation on all config files in the config directory"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Specify categories to evaluate (e.g., benign, harmful)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specify datasets to evaluate (e.g., discord, 4chan)"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        help="Specify detectors to evaluate (e.g., openai_image_checker)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available categories and datasets in the config"
    )
    parser.add_argument(
        "--list-detectors",
        action="store_true",
        help="List available detectors in the config"
    )
    
    args = parser.parse_args()
    
    # If just listing datasets and detectors, do that and return
    if args.list_datasets or args.list_detectors:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        if args.list_datasets:
            print(f"\nModel: {config.get('model_name', 'Unknown')}")
            print("\nAvailable Categories and Datasets:")
            for category, datasets in config["datasets"].items():
                print(f"  Category: {category}")
                for dataset_name, dataset_config in datasets.items():
                    print(f"    - {dataset_name}: {dataset_config['path']}")
                    
        if args.list_detectors:
            print("\nAvailable Detectors:")
            for detector_name, detector_config in config["detectors"].items():
                if detector_config:
                    # If there are parameters, show them
                    params = ", ".join(f"{k}={v}" for k, v in detector_config.items())
                    print(f"  - {detector_name}: {params}")
                else:
                    print(f"  - {detector_name}")
                    
        return
    
    if args.all_configs:
        # Run all config files in the config directory
        config_dir = Path("evaluation_image_detector/config")
        config_files = list(config_dir.glob("*.yaml"))
        
        for config_file in config_files:
            print(f"\n\nRunning evaluation with config: {config_file}")
            runner = ImageBenchmarkRunner(str(config_file))
            
            # Apply filters
            filter_config(runner, args)
                
            runner.run_evaluation()
    else:
        # Run evaluation with specified config
        runner = ImageBenchmarkRunner(args.config)
        
        # Apply filters
        filter_config(runner, args)
        
        all_results = runner.run_evaluation()
        
        # Print high-level results summary
        print("\nEvaluation Results Summary:")
        model = runner.config["model_name"]
        for category, datasets in all_results[model].items():
            print(f"\nCategory: {category}")
            for dataset_name, detector_results in datasets.items():
                print(f"\n  Dataset: {dataset_name}")
                for detector_name, metrics in detector_results.items():
                    if "error" in metrics:
                        print(f"    {detector_name}: Error - {metrics['error']}")
                        continue
                        
                    print(f"\n    {detector_name}:")
                    print("      Basic Metrics:")
                    for metric_name, value in metrics["basic_metrics"].items():
                        print(f"        {metric_name}: {value:.4f}")
                    print(f"      ROC AUC: {metrics['curves']['roc']['auc']:.4f}")
                    print(f"      Average Precision: {metrics['curves']['pr']['average_precision']:.4f}")

def filter_config(runner, args):
    """Apply command line filters to config"""
    
    # Filter categories
    if args.categories:
        filtered_categories = {k: v for k, v in runner.config["datasets"].items() if k in args.categories}
        if not filtered_categories:
            print(f"Warning: None of the specified categories found in config")
            filtered_categories = runner.config["datasets"]
        runner.config["datasets"] = filtered_categories
    
    # Filter datasets (within each category)
    if args.datasets:
        for category in runner.config["datasets"]:
            filtered_datasets = {k: v for k, v in runner.config["datasets"][category].items() if k in args.datasets}
            if not filtered_datasets:
                print(f"Warning: None of the specified datasets found in category '{category}'")
            else:
                runner.config["datasets"][category] = filtered_datasets
    
    # Filter detectors
    if args.detectors:
        filtered_detectors = {k: v for k, v in runner.config["detectors"].items() if k in args.detectors}
        if not filtered_detectors:
            print(f"Warning: None of the specified detectors found in config")
            filtered_detectors = runner.config["detectors"]
        runner.config["detectors"] = filtered_detectors

if __name__ == "__main__":
    main()