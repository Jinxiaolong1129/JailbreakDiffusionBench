import yaml
import json
from pathlib import Path
from typing import Dict, List
import argparse
from jailbreak_diffusion.judger.pre_checker.openai_text_moderation import OpenAITextDetector
from evaluation.data_loader import DatasetLoader
from evaluation.metric import AdvancedMetricsCalculator
from evaluation.visualization import MetricsVisualizer

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
        self.data_loader = DatasetLoader("data")  # 修改为正确的数据目录路径
        self.metrics_calculator = AdvancedMetricsCalculator()
        self.visualizer = MetricsVisualizer(self.output_dir)
        
    def _init_detectors(self) -> Dict:
        """初始化所有检测器"""
        detectors = {}
        detector_configs = self.config["detectors"]
        
        # 只初始化OpenAI检测器
        if "openai" in detector_configs:
            try:
                detectors["openai"] = OpenAITextDetector(**detector_configs["openai"])
            except Exception as e:
                print(f"Error initializing OpenAI detector: {e}")
                    
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
                raw_predictions = detector.check(texts)
                
                predictions = [1 if pred else 0 for pred in raw_predictions]
                # 准备评估数据
                true_labels = [1 if p.label == "harmful" else 0 for p in dataset]  # 修改为harmful标签
                
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
            
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Run OpenAI text detector evaluation")
    parser.add_argument(
        "--config", 
        default="evaluation_text_detector/config/evaluation_configs.yaml",
        help="Path to evaluation config file"
    )
    args = parser.parse_args()
    
    # 运行评估
    runner = BenchmarkRunner(args.config)
    all_results = runner.run_evaluation()
    
    # 打印结果
    print("\nEvaluation Results:")
    for dataset_name, results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        for detector_name, metrics in results.items():
            if "error" in metrics:
                print(f"{detector_name}: Error - {metrics['error']}")
                continue
                
            print(f"\n{detector_name}:")
            print("Basic Metrics:")
            for metric, value in metrics["basic_metrics"].items():
                print(f"  {metric}: {value:.3f}")
            print(f"ROC AUC: {metrics['curves']['roc']['auc']:.3f}")
            print(f"Average Precision: {metrics['curves']['pr']['average_precision']:.3f}")

if __name__ == "__main__":
    main()
