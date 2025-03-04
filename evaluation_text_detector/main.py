import yaml
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple
import argparse
import datetime
import importlib
import sys
import os
import pandas as pd
from collections import defaultdict

from jailbreak_diffusion.judger.pre_checker import OpenAITextDetector, AzureTextDetector, GoogleTextModerator, GPTChecker, LlamaGuardChecker, NSFW_text_classifier_Checker, NSFW_word_match_Checker, distilbert_nsfw_text_checker, distilroberta_nsfw_text_checker, NvidiaAegisChecker

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
        self.output_dir = Path(self.config["output_dir"]) / f"{config_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a copy of the config file used
        with open(self.output_dir / "config_used.yaml", "w") as f:
            yaml.dump(self.config, f)
        
    def initialize_components(self):
        """Initialize all components"""
        self.detectors = self._init_detectors()
        self.data_loader = DatasetLoader(self.config.get("data_dir", ""))
        self.metrics_calculator = TextMetricsCalculator()
        self.visualizer = MetricsVisualizer(str(self.output_dir))
        
    def _init_detectors(self) -> Dict:
        """Initialize all detectors"""
        detectors = {}
        detector_configs = self.config.get("detectors", {})
        
        # Initialize detector mapping
        detector_mapping = {
            "openai_text_moderation": OpenAITextDetector,
            "NSFW_text_classifier": NSFW_text_classifier_Checker,
            "NSFW_word_match": NSFW_word_match_Checker,
            "distilbert_nsfw_text_checker": distilbert_nsfw_text_checker,
            "distilroberta_nsfw_text_checker": distilroberta_nsfw_text_checker,
            "gpt_4o_mini": GPTChecker,
            "gpt_4o": GPTChecker,
            "llama_guard": LlamaGuardChecker,
            "azure_text_moderation": AzureTextDetector,
            "google_text_checker": GoogleTextModerator,
            "nvidia_aegis": NvidiaAegisChecker
        }
        
        for name, config in detector_configs.items():
            if name in detector_mapping:
                try:
                    # Handle detectors that don't need parameters (empty dict)
                    if not config:
                        config = {}
                    
                    detectors[name] = detector_mapping[name](**config)
                    print(f"Successfully initialized {name} detector")
                except Exception as e:
                    print(f"Error initializing {name} detector: {e}")
                    import traceback
                    traceback.print_exc()
                    
        return detectors
    
    def calculate_metrics_by_attribute(self, predictions, attribute_name):
        """Calculate metrics grouped by an attribute (category)"""
        result = {}
        attribute_groups = defaultdict(list)
        
        # Group predictions by the specified attribute
        for pred in predictions:
            attr_value = pred.get(attribute_name, "unknown")
            
            # Handle category being a list
            if isinstance(attr_value, list):
                for value in attr_value:
                    attribute_groups[value].append(pred)
            else:
                attribute_groups[attr_value].append(pred)
                
        # Calculate metrics for each group
        for attr_value, group_preds in attribute_groups.items():
            true_labels = [1 if p["actual_label"] == "harmful" else 0 for p in group_preds]
            pred_scores = [p["prediction_score"] for p in group_preds]
            
            metrics = self.metrics_calculator.calculate_metrics(true_labels, pred_scores)
            
            result[attr_value] = {
                "count": len(group_preds),
                "metrics": metrics["basic_metrics"],
                "confusion_matrix": metrics["confusion_matrix"]
            }
            
        return {
            "total_count": sum(len(group) for group in attribute_groups.values()),
            "groups": list(attribute_groups.keys()),
            "metrics": result
        }
        
    def evaluate_dataset(self, dataset_path: str) -> Dict:
        """Evaluate a single dataset with all detectors"""
        print(f"\nEvaluating dataset: {dataset_path}")
        
        # Create dataset specific output directory
        dataset_name = Path(dataset_path).stem
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualizer output directory
        self.visualizer.output_dir = dataset_output_dir
        
        # Load dataset
        dataset = self.data_loader.load_dataset(dataset_path)
        dataset_info = self.data_loader.get_dataset_info(dataset_path)
        print(f"Dataset info: {dataset_info}")
        
        # Save dataset info and original dataset JSON content
        with open(dataset_output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
            
        # Copy the original dataset JSON file for reference
        with open(dataset_path, "r") as src_f:
            original_dataset = json.load(src_f)
            data_filename = Path(dataset_path).name
            with open(dataset_output_dir / data_filename, "w") as dst_f:
                json.dump(original_dataset, dst_f, indent=2)
        
        results = {}
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name} on {dataset_path}...")
            
            try:
                # Get detection results
                texts = [prompt.text for prompt in dataset]
                
                batch_size = self.config.get("batch_size", 32)
                
                raw_predictions = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    # Request scores for better metrics calculation
                    batch_preds = detector.check(batch_texts, return_scores=True)
                    raw_predictions.extend(batch_preds)
                    print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

                if len(raw_predictions) != len(texts):
                    print(f"ERROR: raw_predictions length ({len(raw_predictions)}) doesn't match texts length ({len(texts)})")
                    while len(raw_predictions) < len(texts):
                        raw_predictions.append({'flagged': False, 'score': 0.0})
                    raw_predictions = raw_predictions[:len(texts)]

                # Extract binary predictions and confidence scores for metrics calculation
                binary_predictions = []
                confidence_scores = []

                for pred in raw_predictions:
                    if pred is None:
                        binary_predictions.append(0)
                        confidence_scores.append(0.0)
                    elif isinstance(pred, bool):
                        binary_predictions.append(1 if pred else 0)
                        confidence_scores.append(1.0 if pred else 0.0)
                    elif isinstance(pred, dict):
                        # Handle dictionary format (detailed results with scores)
                        if "flagged" in pred:
                            binary_predictions.append(1 if pred["flagged"] else 0)
                            if "score" in pred:
                                if "label" in pred:
                                    confidence_scores.append(float(pred["score"]))
                            else:
                                confidence_scores.append(1.0 if pred["flagged"] else 0.0)
                        elif "harmful" in pred:
                            binary_predictions.append(1 if pred["harmful"] else 0)
                            if "score" in pred:
                                confidence_scores.append(float(pred["score"]))
                            else:
                                confidence_scores.append(1.0 if pred["harmful"] else 0.0)
                        elif "score" in pred:
                            score = float(pred["score"])
                            binary_predictions.append(1 if score >= 0.5 else 0)
                            confidence_scores.append(score)
                        else:
                            print(f"Warning: Unknown dict format in prediction: {pred.keys()}")
                            binary_predictions.append(0)
                            confidence_scores.append(0.0)
                    elif isinstance(pred, (int, float)):
                        # Handle simple numeric score
                        try:
                            score = float(pred)
                            if 0 <= score <= 1:
                                binary_predictions.append(1 if score >= 0.5 else 0)
                                confidence_scores.append(score)
                            else:
                                print(f"Warning: Score out of range [0,1]: {score}")
                                binary_predictions.append(0)
                                confidence_scores.append(0.5)
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {pred} to float")
                            binary_predictions.append(0)
                            confidence_scores.append(0.5)
                    else:
                        print(f"Warning: Unknown prediction type: {type(pred)}")
                        binary_predictions.append(0)
                        confidence_scores.append(0.0)

                # Prepare evaluation data
                true_labels = [1 if p.label == "harmful" else 0 for p in dataset]
                
                # Calculate metrics using the confidence scores for better ROC/PR curves
                metrics = self.metrics_calculator.calculate_metrics(true_labels, confidence_scores)
                
                # Double-check accuracy calculation
                correct_count = sum(1 for i in range(len(binary_predictions)) if binary_predictions[i] == true_labels[i])
                calculated_accuracy = correct_count / len(binary_predictions)
                
                # If there's a significant discrepancy, override the calculated accuracy
                if abs(metrics["basic_metrics"]["accuracy"] - calculated_accuracy) > 0.01:
                    print(f"Warning: Original accuracy calculation ({metrics['basic_metrics']['accuracy']:.4f}) differs from direct calculation ({calculated_accuracy:.4f})")
                    print(f"Using direct calculation instead: {correct_count} correct predictions out of {len(binary_predictions)} total")
                    metrics["basic_metrics"]["accuracy"] = calculated_accuracy
                    
                results[detector_name] = metrics
                
                detailed_predictions = []
                for i, (prompt, binary_pred, score, true_label) in enumerate(zip(dataset, binary_predictions, confidence_scores, true_labels)):
                    detailed_predictions.append({
                        "id": prompt.id,
                        "text": prompt.text,
                        "actual_label": "harmful" if true_label == 1 else "benign",
                        "predicted_label": "harmful" if binary_pred == 1 else "benign",
                        "prediction_score": float(score),  # Use the extracted confidence score
                        "correct": binary_pred == true_label,
                        "category": prompt.category,
                        "source": prompt.source,
                        "raw_prediction": raw_predictions[i]
                    })

                # Calculate metrics by category only
                category_metrics = self.calculate_metrics_by_attribute(detailed_predictions, "category")
                
                # Add category breakdown to results
                results[detector_name]["category_breakdown"] = category_metrics
                
                # Save detailed results
                detailed_results = {
                    "detector": detector_name,
                    "dataset": dataset_name,
                    "total_prompts": len(detailed_predictions),
                    "correctly_classified": sum(1 for p in detailed_predictions if p["correct"]),
                    "metrics": metrics["basic_metrics"],
                    "category_breakdown": category_metrics,
                    "predictions": detailed_predictions
                }
                
                # Save detector results to its own file
                detector_output_file = dataset_output_dir / f"{dataset_name}_{detector_name}_results.json"
                detailed_output_file = dataset_output_dir / f"{dataset_name}_{detector_name}_detailed_predictions.json"
                
                with open(detector_output_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
                with open(detailed_output_file, "w") as f:
                    json.dump(detailed_results, f, indent=2)
                
                # Save metrics by category to separate file
                category_metrics_file = dataset_output_dir / f"{dataset_name}_{detector_name}_category_metrics.json"
                with open(category_metrics_file, "w") as f:
                    json.dump(category_metrics, f, indent=2)
                
                # Save misclassified examples if configured
                if self.config.get("save_misclassified", False):
                    misclassified = [p for p in detailed_predictions if not p["correct"]]
                    if misclassified:
                        misclassified_file = dataset_output_dir / f"{dataset_name}_{detector_name}_misclassified.json"
                        with open(misclassified_file, "w") as f:
                            json.dump(misclassified, f, indent=2)
                
                # Create visualizations for category breakdowns
                try:
                    # Visualize performance by category
                    self.visualizer.plot_category_performance(
                        {"categories": category_metrics["groups"], "metrics": category_metrics["metrics"]},
                        f"{dataset_name}_{detector_name}_Category_Performance"
                    )
                except Exception as e:
                    print(f"Error generating category breakdown visualizations: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"Error evaluating {detector_name} on {dataset_path}: {e}")
                import traceback
                traceback.print_exc()
                results[detector_name] = {"error": str(e)}
                
        # Generate visualizations for this dataset
        try:
            self.visualizer.plot_roc_curves(results, f"{dataset_name} ROC Curves")
            self.visualizer.plot_pr_curves(results, f"{dataset_name} PR Curves")
            self.visualizer.plot_confusion_matrices(results, f"{dataset_name} Confusion Matrices")
            self.visualizer.plot_f1_threshold_curves(results, f"{dataset_name} F1 vs Threshold")
        except Exception as e:
            print(f"Error generating visualizations for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
                
        return results
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        dataset_path = self.config.get("datasets", {}).get("path", "")
        
        if not dataset_path:
            print("Error: No dataset path specified in config")
            return {}
            
        results = self.evaluate_dataset(dataset_path)
        
        # Create overall summary
        self._create_experiment_summary(results)
        
        return results
    
    def _create_experiment_summary(self, results: Dict):
        """Create a summary of the experiment across all detectors"""
        summary = {
            "config_file": self.config_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": self.config.get("datasets", {}).get("path", ""),
            "detectors": list(self.config.get("detectors", {}).keys()),
            "results_summary": {},
            "category_breakdown_summary": {}
        }
        
        # Create a summary of key metrics for all detectors
        for detector_name, metrics in results.items():
            if "error" in metrics:
                summary["results_summary"][detector_name] = {
                    "error": metrics["error"]
                }
                continue
                
            # Extract key metrics
            summary["results_summary"][detector_name] = {
                "accuracy": metrics["basic_metrics"]["accuracy"],
                "precision": metrics["basic_metrics"]["precision"],
                "recall": metrics["basic_metrics"]["recall"],
                "f1": metrics["basic_metrics"].get("f1", 0.0),
                "roc_auc": metrics["curves"]["roc"]["auc"],
                "average_precision": metrics["curves"]["pr"]["average_precision"]
            }
                    
            # Add category breakdown
            if "category_breakdown" in metrics:
                summary["category_breakdown_summary"][detector_name] = {}
                for category, category_metrics in metrics["category_breakdown"]["metrics"].items():
                    summary["category_breakdown_summary"][detector_name][category] = {
                        "count": category_metrics["count"],
                        "accuracy": category_metrics["metrics"]["accuracy"],
                        "precision": category_metrics["metrics"]["precision"],
                        "recall": category_metrics["metrics"]["recall"],
                        "f1": category_metrics["metrics"].get("f1", 0.0)
                    }
        
        # Save summary to file
        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate summary tables as text
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            dataset_name = Path(summary["dataset"]).stem
            f.write(f"Experiment Summary\n")
            f.write(f"=================\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Date: {summary['timestamp']}\n")
            f.write(f"Dataset: {summary['dataset']}\n\n")
            
            # Create header for overall metrics
            f.write(f"Overall Metrics\n")
            f.write(f"==============\n")
            metrics_header = ["Detector", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"]
            f.write(" | ".join(metrics_header) + "\n")
            f.write("-" * 80 + "\n")
            
            # Add data rows
            for detector_name in summary["detectors"]:
                if detector_name in summary["results_summary"]:
                    detector_metrics = summary["results_summary"][detector_name]
                    
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
            
            # Add category breakdown summary
            if summary["category_breakdown_summary"]:
                f.write(f"Metrics by Category\n")
                f.write(f"=================\n\n")
                
                for detector_name, categories in summary["category_breakdown_summary"].items():
                    f.write(f"Detector: {detector_name}\n")
                    f.write("-" * (len(detector_name) + 10) + "\n")
                    
                    # Create header
                    metrics_header = ["Category", "Count", "Accuracy", "Precision", "Recall", "F1"]
                    f.write(" | ".join(metrics_header) + "\n")
                    f.write("-" * 80 + "\n")
                    
                    # Add category rows
                    for category, metrics in categories.items():
                        row = [
                            category,
                            f"{metrics['count']}",
                            f"{metrics['accuracy']:.4f}",
                            f"{metrics['precision']:.4f}",
                            f"{metrics['recall']:.4f}",
                            f"{metrics['f1']:.4f}"
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
        "--dataset",
        help="Override dataset path in config"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        help="Specify detectors to evaluate (e.g., openai_text_moderation, llama_guard)"
    )
    
    args = parser.parse_args()
    
    # Initialize the benchmark runner
    runner = TextBenchmarkRunner(args.config)
    
    # Override dataset path if provided
    if args.dataset:
        runner.config["datasets"]["path"] = args.dataset
    
    # Filter detectors if specified
    if args.detectors:
        filtered_detectors = {k: v for k, v in runner.config.get("detectors", {}).items() if k in args.detectors}
        if not filtered_detectors:
            print(f"Warning: None of the specified detectors found in config")
        else:
            runner.config["detectors"] = filtered_detectors
    
    # Run the evaluation
    results = runner.run_evaluation()
    
    # Print high-level results summary
    print("\nEvaluation Results Summary:")
    dataset_name = Path(runner.config.get("datasets", {}).get("path", "")).stem
    print(f"\nDataset: {dataset_name}")
    
    for detector_name, metrics in results.items():
        if "error" in metrics:
            print(f"  {detector_name}: Error - {metrics['error']}")
            continue
            
        print(f"\n  {detector_name}:")
        print("    Basic Metrics:")
        for metric_name, value in metrics["basic_metrics"].items():
            print(f"      {metric_name}: {value:.4f}")
        print(f"    ROC AUC: {metrics['curves']['roc']['auc']:.4f}")
        print(f"    Average Precision: {metrics['curves']['pr']['average_precision']:.4f}")
        
        # Print category breakdown
        if "category_breakdown" in metrics:
            print("\n    Metrics by Category:")
            for category, category_metrics in metrics["category_breakdown"]["metrics"].items():
                print(f"      {category} (n={category_metrics['count']}):")
                print(f"        Accuracy: {category_metrics['metrics']['accuracy']:.4f}")
                print(f"        F1: {category_metrics['metrics'].get('f1', 0.0):.4f}")


if __name__ == "__main__":
    main()
    
    
    
# PYTHONPATH=/home/jin509/jailbreak_diffusion_benchmark/JailbreakDiffusionBench python evaluation_text_detector/main.py --config evaluation_text_detector/config/gpt_4o_mini.yaml 


# PYTHONPATH=/home/jin509/jailbreak_diffusion_benchmark/JailbreakDiffusionBench python evaluation_text_detector/main.py --config evaluation_text_detector/config/gpt_4o.yaml 