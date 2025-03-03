# evaluation_text_detector/main.py
import yaml
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple
import argparse
import datetime
import importlib
import sys
import argparse
from pathlib import Path
import yaml
import sys
import os

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
        
        # Create category directories (benign/harmful)
        for category in self.config["datasets"].keys():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
        
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
        
        # Initialize detector mapping
        detector_mapping = {
            "openai_text_moderation": OpenAITextDetector,
            "NSFW_text_classifier": NSFW_text_classifier_Checker,
            "NSFW_word_match": NSFW_word_match_Checker,
            "distilbert_nsfw_text_checker": distilbert_nsfw_text_checker,
            "distilroberta_nsfw_text_checker": distilroberta_nsfw_text_checker,
            "gpt_4o_mini": GPTChecker,
            "llama_guard": LlamaGuardChecker,
            "azure_text_moderation": AzureTextDetector,
            "google_text_moderation": GoogleTextModerator,
            "nvidia_aegis": NvidiaAegisChecker
        }
        
        for name, config in detector_configs.items():
            if name in detector_mapping:
                try:
                    # 处理不需要参数的检测器（空字典）
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
        将嵌套的数据集结构转换为展平的列表
        返回: [(category, dataset_name, dataset_config), ...]
        """
        flattened_datasets = []
        for category, datasets in self.config["datasets"].items():
            for dataset_name, dataset_config in datasets.items():
                flattened_datasets.append((category, dataset_name, dataset_config))
        return flattened_datasets
        
    def evaluate_dataset(self, category: str, dataset_name: str, dataset_config: Dict) -> Dict:
        """Evaluate a single dataset with all detectors"""
        print(f"\nEvaluating dataset: {category}/{dataset_name}")
        
        # Create dataset specific output directory (under the category)
        dataset_output_dir = self.output_dir / category / dataset_name
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
            
        # Copy the original dataset JSON file for reference
        with open(dataset_config["path"], "r") as src_f:
            original_dataset = json.load(src_f)
            dataset_path = Path(dataset_config["path"])
            data_filename = dataset_path.name
            data_name = dataset_path.stem
            print(f"Extracted filename: {data_filename}")
            with open(dataset_output_dir / data_filename, "w") as dst_f:
                json.dump(original_dataset, dst_f, indent=2)
        
        results = {}
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name} on {category}/{dataset_name}...")
            
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
                                if pred["label"] == "NSFW":
                                    # For NSFW label, use score directly
                                    confidence_scores.append(float(pred["score"]))
                                else:
                                    # For SFW label, use appropriate score for AUC calculation
                                    # We need probability of NSFW class for ROC calculation
                                    # This depends on the classifier's output format
                                    # When using transformers, we're getting the probability of the predicted class
                                    # Binary classifiers typically need the probability of the positive class
                                    confidence_scores.append(1.0 - float(pred["score"]))
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
                categories = [p.category for p in dataset]

                # Calculate metrics using the confidence scores for better ROC/PR curves
                metrics = self.metrics_calculator.calculate_metrics(true_labels, confidence_scores)
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

                # Save detailed results
                detailed_results = {
                    "detector": detector_name,
                    "dataset": f"{category}/{dataset_name}",
                    "total_prompts": len(detailed_predictions),
                    "correctly_classified": sum(1 for p in detailed_predictions if p["correct"]),
                    "metrics": metrics["basic_metrics"],
                    "predictions": detailed_predictions
                }
                
                # Save detector results to its own file
                detector_output_file = dataset_output_dir / f"{data_name}_{detector_name}_results.json"
                detailed_output_file = dataset_output_dir / f"{data_name}_{detector_name}_detailed_predictions.json"
                
                with open(detector_output_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
                with open(detailed_output_file, "w") as f:
                    json.dump(detailed_results, f, indent=2)
                
                
            except Exception as e:
                print(f"Error evaluating {detector_name} on {category}/{dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[detector_name] = {"error": str(e)}
                
        # Generate visualizations for this dataset
        try:
            self.visualizer.plot_roc_curves(results, f"{dataset_name} ROC Curves")
            self.visualizer.plot_pr_curves(results, f"{dataset_name} PR Curves")
        except Exception as e:
            print(f"Error generating visualizations for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
                
        return results
    
    
    def run_evaluation(self):
        """Run complete evaluation pipeline for all datasets and detectors"""
        all_results = {}
        
        # 获取展平的数据集列表
        flattened_datasets = self._get_flattened_datasets()
        
        # 按类别/数据集组织结果
        for category, dataset_name, dataset_config in flattened_datasets:
            # 确保类别存在于结果字典中
            if category not in all_results:
                all_results[category] = {}
                
            # 运行评估
            dataset_results = self.evaluate_dataset(category, dataset_name, dataset_config)
            all_results[category][dataset_name] = dataset_results
        
        # Create overall summary
        self._create_experiment_summary(all_results)
        
        return all_results
    
    def _create_experiment_summary(self, all_results: Dict):
        """Create a summary of the experiment across all datasets and detectors"""
        summary = {
            "config_file": self.config_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "categories": list(self.config["datasets"].keys()),
            "detectors": list(self.config["detectors"].keys()),
            "results_summary": {}
        }
        
        # Create a summary table of key metrics for all detectors across all datasets
        for category, datasets in all_results.items():
            summary["results_summary"][category] = {}
            
            for dataset_name, detector_results in datasets.items():
                summary["results_summary"][category][dataset_name] = {}
                
                for detector_name, metrics in detector_results.items():
                    if "error" in metrics:
                        summary["results_summary"][category][dataset_name][detector_name] = {
                            "error": metrics["error"]
                        }
                        continue
                        
                    # Extract key metrics
                    summary["results_summary"][category][dataset_name][detector_name] = {
                        "accuracy": metrics["basic_metrics"]["accuracy"],
                        "precision": metrics["basic_metrics"]["precision"],
                        "recall": metrics["basic_metrics"]["recall"],
                        "f1": metrics["basic_metrics"].get("f1", 0.0),
                        "roc_auc": metrics["curves"]["roc"]["auc"],
                        "average_precision": metrics["curves"]["pr"]["average_precision"]
                    }
        
        # 为每个类别计算平均指标
        category_averages = {}
        for category, datasets in summary["results_summary"].items():
            category_averages[category] = {}
            
            for detector_name in summary["detectors"]:
                detector_metrics = []
                
                for dataset_name, detector_results in datasets.items():
                    if detector_name in detector_results and "error" not in detector_results[detector_name]:
                        detector_metrics.append(detector_results[detector_name])
                
                if detector_metrics:
                    # 计算平均值
                    category_averages[category][detector_name] = {
                        "accuracy": sum(m["accuracy"] for m in detector_metrics) / len(detector_metrics),
                        "precision": sum(m["precision"] for m in detector_metrics) / len(detector_metrics),
                        "recall": sum(m["recall"] for m in detector_metrics) / len(detector_metrics),
                        "f1": sum(m["f1"] for m in detector_metrics) / len(detector_metrics),
                        "roc_auc": sum(m["roc_auc"] for m in detector_metrics) / len(detector_metrics),
                        "average_precision": sum(m["average_precision"] for m in detector_metrics) / len(detector_metrics)
                    }
        
        # 添加类别平均指标到摘要
        summary["category_averages"] = category_averages
        
        # Save summary to file
        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate summary tables as text
        with open(self.output_dir / "experiment_summary.txt", "w") as f:
            f.write(f"Experiment Summary\n")
            f.write(f"=================\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Date: {summary['timestamp']}\n\n")
            
            for category in summary["categories"]:
                f.write(f"\nCategory: {category}\n")
                f.write("=" * (len(category) + 10) + "\n\n")
                
                # 对每个类别下的数据集
                for dataset_name in summary["results_summary"][category].keys():
                    f.write(f"\nDataset: {dataset_name}\n")
                    f.write("-" * (len(dataset_name) + 10) + "\n")
                    
                    # Create header
                    metrics_header = ["Detector", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "AP"]
                    f.write(" | ".join(metrics_header) + "\n")
                    f.write("-" * 80 + "\n")
                    
                    # Add data rows
                    for detector_name in summary["detectors"]:
                        if detector_name in summary["results_summary"][category][dataset_name]:
                            detector_metrics = summary["results_summary"][category][dataset_name][detector_name]
                            
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
                
                # 打印类别平均指标
                f.write(f"\nCategory Average: {category}\n")
                f.write("-" * (len(category) + 17) + "\n")
                
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
                
                
                            
def main():
    parser = argparse.ArgumentParser(description="Run text detector evaluation")
    parser.add_argument(
        "--config", 
        default="evaluation_text_detector/config/openai_text_moderation.yaml",
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
        help="Specify categories to evaluate (e.g., harmful, benign)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specify datasets to evaluate (e.g., i2p, civitai)"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        help="Specify detectors to evaluate (e.g., openai_text_moderation, llama_guard)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets in the config"
    )
    parser.add_argument(
        "--list-detectors",
        action="store_true",
        help="List available detectors in the config"
    )
    
    args = parser.parse_args()
    
    # 如果只是列出数据集和检测器，处理后直接返回
    if args.list_datasets or args.list_detectors:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            
        if args.list_datasets:
            print("\nAvailable Categories and Datasets:")
            for category, datasets in config["datasets"].items():
                print(f"\n{category.upper()}:")
                for dataset_name, dataset_config in datasets.items():
                    print(f"  - {dataset_name}: {dataset_config['path']}")
                    
        if args.list_detectors:
            print("\nAvailable Detectors:")
            for detector_name, detector_config in config["detectors"].items():
                if detector_config:
                    # 如果有参数，显示参数
                    params = ", ".join(f"{k}={v}" for k, v in detector_config.items())
                    print(f"  - {detector_name}: {params}")
                else:
                    print(f"  - {detector_name}")
                    
        return
    
    if args.all_configs:
        # Run all config files in the config directory
        config_dir = Path("evaluation_text_detector/config")
        config_files = list(config_dir.glob("*.yaml"))
        
        for config_file in config_files:
            print(f"\n\nRunning evaluation with config: {config_file}")
            runner = TextBenchmarkRunner(str(config_file))
            
            # 应用过滤条件
            filter_config(runner, args)
                
            runner.run_evaluation()
    else:
        # Run evaluation with specified config
        runner = TextBenchmarkRunner(args.config)
        
        # 应用过滤条件
        filter_config(runner, args)
        
        all_results = runner.run_evaluation()
        
        # Print high-level results summary
        print("\nEvaluation Results Summary:")
        for category, datasets in all_results.items():
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
    """应用命令行参数过滤配置"""
    
    # 过滤类别
    if args.categories:
        filtered_categories = {k: v for k, v in runner.config["datasets"].items() if k in args.categories}
        if not filtered_categories:
            print(f"Warning: None of the specified categories found in config")
            filtered_categories = runner.config["datasets"]
        runner.config["datasets"] = filtered_categories
    
    # 过滤数据集（在每个类别内）
    if args.datasets:
        for category in runner.config["datasets"]:
            filtered_datasets = {k: v for k, v in runner.config["datasets"][category].items() if k in args.datasets}
            if not filtered_datasets:
                print(f"Warning: None of the specified datasets found in category '{category}'")
            else:
                runner.config["datasets"][category] = filtered_datasets
    
    # 过滤检测器
    if args.detectors:
        filtered_detectors = {k: v for k, v in runner.config["detectors"].items() if k in args.detectors}
        if not filtered_detectors:
            print(f"Warning: None of the specified detectors found in config")
            filtered_detectors = runner.config["detectors"]
        runner.config["detectors"] = filtered_detectors

if __name__ == "__main__":
    main()