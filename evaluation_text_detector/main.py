import yaml
import json
from pathlib import Path
from typing import Dict, List
import argparse
from jailbreak_diffusion.judger.pre_checker import (
    AzureTextDetector,
    NSFWTextDetector,
    LlamaGuardDetector,
    GPTDetector
)
from evaluation.data_loader import DatasetLoader
from evaluation.metrics import AdvancedMetricsCalculator
from evaluation.visualizations import MetricsVisualizer

class BenchmarkRunner:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.setup_directories()
        self.initialize_components()
        
    def load_config(self):
        """加载配置文件"""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
    def setup_directories(self):
        """创建必要的目录"""
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_components(self):
        """初始化各个组件"""
        self.detectors = self._init_detectors()
        self.data_loader = DatasetLoader(self.config["data_dir"])
        self.metrics_calculator = AdvancedMetricsCalculator()
        self.visualizer = MetricsVisualizer(self.output_dir)
        
    def _init_detectors(self) -> Dict:
        """初始化所有检测器"""
        detectors = {}
        detector_configs = self.config["detectors"]
        
        detector_mapping = {
            "azure": AzureTextDetector,
            "nsfw": NSFWTextDetector,
            "llama_guard": LlamaGuardDetector,
            "gpt": GPTDetector
        }
        
        for name, config in detector_configs.items():
            if name in detector_mapping:
                try:
                    detectors[name] = detector_mapping[name](**config)
                except Exception as e:
                    print(f"Error initializing {name} detector: {e}")
                    
        return detectors
        
    def evaluate_dataset(self, dataset_name: str, dataset_config: Dict) -> Dict:
        """评估单个数据集"""
        print(f"\nEvaluating dataset: {dataset_name}")
        
        # 加载数据集
        dataset = self.data_loader.load_dataset(dataset_config["path"])
        dataset_info = self.data_loader.get_dataset_info(dataset_config["path"])
        print(f"Dataset info: {dataset_info}")
        
        results = {}
        for detector_name, detector in self.detectors.items():
            print(f"Running {detector_name}...")
            
            try:
                # 获取检测结果
                texts = [prompt.text for prompt in dataset]
                predictions = detector.check(texts)
                
                # 准备评估数据
                true_labels = [1 if p.label == "malicious" else 0 for p in dataset]
                
                # 计算指标
                metrics = self.metrics_calculator.calculate_metrics(true_labels, predictions)
                results[detector_name] = metrics
                
            except Exception as e:
                print(f"Error evaluating {detector_name}: {e}")
                results[detector_name] = {"error": str(e)}
                
        return results
                
    def run_evaluation(self):
        """运行完整评估流程"""
        all_results = {}
        
        # 评估每个数据集
        for dataset_name, dataset_config in self.config["datasets"].items():
            results = self.evaluate_dataset(dataset_name, dataset_config)
            all_results[dataset_name] = results
            
            # 保存结果
            output_file = self.output_dir / f"{dataset_name}_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # 生成该数据集的可视化
            self.visualizer.plot_roc_curves(results, f"{dataset_name} ROC Curves")
            self.visualizer.plot_pr_curves(results, f"{dataset_name} PR Curves")
            self.visualizer.plot_metrics_heatmap(results, f"{dataset_name} Metrics Comparison")
            
        # 计算加权平均结果
        overall_results = self.calculate_overall_results(all_results)
        
        # 保存总体结果
        with open(self.output_dir / "overall_results.json", "w") as f:
            json.dump(overall_results, f, indent=2)
            
        # 生成总体可视化
        self.visualizer.plot_metrics_heatmap(
            overall_results, 
            "Overall Metrics Comparison"
        )
        
        return all_results, overall_results
    
    def calculate_overall_results(self, all_results: Dict) -> Dict:
        """计算所有数据集的加权平均结果"""
        overall_results = {}
        
        for detector_name in self.detectors.keys():
            detector_metrics = {
                "basic_metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "false_positive_rate": 0.0,
                    "false_negative_rate": 0.0
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
                
                # 累积基础指标
                for metric in detector_metrics["basic_metrics"]:
                    detector_metrics["basic_metrics"][metric] += (
                        results[detector_name]["basic_metrics"][metric] * weight
                    )
                
                # 累积曲线指标
                detector_metrics["curves"]["roc"]["auc"] += (
                    results[detector_name]["curves"]["roc"]["auc"] * weight
                )
                detector_metrics["curves"]["pr"]["average_precision"] += (
                    results[detector_name]["curves"]["pr"]["average_precision"] * weight
                )
            
            # 计算加权平均
            if total_weight > 0:
                for metric in detector_metrics["basic_metrics"]:
                    detector_metrics["basic_metrics"][metric] /= total_weight
                    
                detector_metrics["curves"]["roc"]["auc"] /= total_weight
                detector_metrics["curves"]["pr"]["average_precision"] /= total_weight
                
            overall_results[detector_name] = detector_metrics
            
        return overall_results

def main():
    parser = argparse.ArgumentParser(description="Run text detector evaluation")
    parser.add_argument(
        "--config", 
        default="config/evaluation_configs.yaml",
        help="Path to evaluation config file"
    )
    args = parser.parse_args()
    
    # 运行评估
    runner = BenchmarkRunner(args.config)
    all_results, overall_results = runner.run_evaluation()
    
    # 打印总体结果
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