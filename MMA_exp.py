import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
import yaml
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from MMA import ParallelMMA

@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]

@dataclass
class BenchmarkConfig:
    experiment_name: str
    output_dir: str
    model: ModelConfig
    datasets: List[str]
    attack_method: str
    log_level: str = "INFO"
    batch_size: int = 32
    optimization_steps: int = 1000
    topk: int = 256
    save_history: bool = True
    save_intermediate: bool = False
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # 转换model配置
        config_dict['model'] = ModelConfig(**config_dict['model'])
        return cls(**config_dict)




class PromptDataset:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'index': row['index'],
            'prompt': row['prompt']
        }

class ParallelBenchmark:
    def __init__(
        self,
        config: BenchmarkConfig,
        dataset_path: str,
        text_encoder,
        tokenizer,
    ):
        self.config = config
        self.dataset_path = dataset_path
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        # Setup dataset
        self.dataset = PromptDataset(dataset_path)
        
        # Create experiment directory structure
        self.exp_dir = self._setup_experiment_dir()
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        
        # Initialize results list
        self.results = []
        
    def _setup_experiment_dir(self) -> Path:
        """Create and return experiment directory"""
        exp_dir = Path(self.config.output_dir) / \
                 "MMA" / \
                 self.config.model.name / \
                 Path(self.dataset_path).stem
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _setup_logging(self):
        """Configure logging"""
        exp_name = f"MMA_{self.config.model.name}_{Path(self.dataset_path).stem}"
        
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(self.config.log_level)
        
        # File handler
        log_file = self.exp_dir / f"{exp_name}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(self.config.log_level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(self.config.log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info(f"Logging file: {log_file}")

    def _process_results(self, attack_results: List[Dict], prompts: List[Dict]) -> List[Dict]:
        """处理攻击结果"""
        processed_results = []
        for result, prompt in zip(attack_results, prompts):
            processed = {
                "prompt_index": prompt['index'],
                "original_prompt": prompt['prompt'],
                "attack_prompt": result.attack_prompt,
                "success": result.success,
                "execution_time": result.execution_time
            }
            processed_results.append(processed)
        return processed_results

    def _save_results(self):
        """保存实验结果"""
        # 保存详细结果
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.exp_dir / "results.csv", index=False)
        
        # 保存实验元数据
        metadata = {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": str(self.dataset.csv_path),
            "num_prompts": len(self.dataset),
            "model_name": self.config.model.name,
        }
        
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # 保存每个prompt的优化历史
        history_dir = self.exp_dir / "optimization_history"
        history_dir.mkdir(exist_ok=True)
        
        # tracker的历史记录会自动保存到相应目录

    def run(self, batch_size: int = 32):
        """运行实验"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Model: {self.config.model.name}")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Processing {len(self.dataset)} prompts")
        
        try:
            # 批量处理prompts
            prompts = [self.dataset[i] for i in range(len(self.dataset))]
            prompt_texts = [p['prompt'] for p in prompts]
            
            # 初始化ParallelMMA
            mma = ParallelMMA(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                experiment_name=f"MMA_{self.config.model.name}_{Path(self.dataset_path).stem}",
                save_dir=str(self.exp_dir / "optimization_history")
            )
            
            # 批量攻击
            results = mma.attack_batch_parallel(
                prompts=prompt_texts,
                batch_size=batch_size
            )
            
            # 处理结果
            processed_results = self._process_results(results, prompts)
            self.results.extend(processed_results)
            
            # 保存结果
            self._save_results()
            
            self.logger.info("Experiment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
            
    def get_summary(self) -> Dict:
        """返回实验统计摘要"""
        if not self.results:
            return {}
            
        results_df = pd.DataFrame(self.results)
        
        summary = {
            "model": self.config.model.name,
            "dataset": Path(self.dataset_path).stem,
            "attack_method": "MMA",
            "total_prompts": len(results_df),
            "avg_execution_time": results_df["execution_time"].mean(),
        }
        
        return summary

def run_parallel_benchmark(config_path: str, text_encoder, tokenizer):
    """运行基准测试"""
    config = BenchmarkConfig.from_yaml(config_path)
    
    all_summaries = []
    
    for dataset_path in config.datasets:
        try:
            experiment = ParallelBenchmark(
                config=config,
                dataset_path=dataset_path,
                text_encoder=text_encoder,
                tokenizer=tokenizer
            )
            experiment.run()
            summary = experiment.get_summary()
            all_summaries.append(summary)
        except Exception as e:
            print(f"Error running experiment with dataset={dataset_path}: {str(e)}")
            raise
    
    # 保存总体摘要
    summary_df = pd.DataFrame(all_summaries)
    summary_path = Path(config.output_dir) / \
                "MMA" / \
                config.model.name / \
                f"MMA_{config.model.name}_summary.csv"
    
    os.makedirs(summary_path.parent, exist_ok=True)
    
    if not summary_path.exists():
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)

    print(f"Overall summary saved to {summary_path}")




# 使用示例：
"""
import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 加载模型
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# 运行实验
run_parallel_benchmark(
    config_path='config/benchmark_config.yaml',
    text_encoder=text_encoder,
    tokenizer=tokenizer
)
"""


if __name__ == "__main__":
    # 加载模型
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 运行实验
    run_parallel_benchmark(
        config_path='benchmark_config/MMA/stable-diffusion-3.5-medium_MMA.yaml',
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )