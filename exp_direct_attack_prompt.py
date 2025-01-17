import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import json
from tqdm import tqdm
import yaml
from jailbreak_diffusion.attack import AttackerFactory
from jailbreak_diffusion.diffusion_model import DiffusionFactory


@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    params: Dict[str, Any]

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments"""
    experiment_name: str
    output_dir: str
    save_images: bool
    save_prompts: bool
    log_level: str
    batch_size: int
    num_workers: int
    model: ModelConfig
    attack_method: str
    datasets: List[str]
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BenchmarkConfig':
        """Create config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert model dict to ModelConfig
        model_dict = config_dict.pop('model')
        model_config = ModelConfig(**model_dict)
        
        return cls(model=model_config, **config_dict)



class PromptDataset:
    """Dataset class for loading and managing prompts with attack prompts"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self._validate_data()
        
    def _validate_data(self):
        required_cols = ['prompt_index', 'original_prompt', 'attack_prompt']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'index': row['prompt_index'],
            'original_prompt': row['original_prompt'],
            'attack_prompt': row['attack_prompt']
        }



class BenchmarkExperiment:
    def __init__(
        self,
        config: BenchmarkConfig,
        dataset_path: str,
    ):
        self.config = config
        self.dataset_path = dataset_path
        
        # Setup model and attacker
        self.model = DiffusionFactory(
            **self.config.model.params
        )
        self.attacker = AttackerFactory(
            self.config.attack_method,
            target_model=self.model
        )
        
        # Setup dataset
        self.dataset = PromptDataset(dataset_path)
        
        # Create base output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create experiment directory
        self.exp_dir = self._setup_experiment_dir()
        self.image_dir = self.exp_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        
        self.results = []
        
        
        
    def _setup_experiment_dir(self) -> Path:
        """Create and return experiment directory with new structure"""
        # 新的目录结构：benchmark_results/[attack_method]/[model_name]/[dataset_name]
        exp_dir = Path(self.config.output_dir) / \
                 self.config.attack_method / \
                 self.config.model.name / \
                 Path(self.dataset_path).stem
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _setup_logging(self):
        """Configure logging"""
        # 修改日志文件路径以匹配新的目录结构
        exp_name = f"{self.config.attack_method}_{self.config.model.name}_{Path(self.dataset_path).stem}"
        
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(self.config.log_level)
        
        # File handler - 使用新的目录结构
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

    def _save_image(self, image: Any, prompt_index: str, save=True) -> str:
        """Save generated image and return path"""
        if isinstance(image, list):
            if len(image) > 1:
                raise ValueError("Image list contains more than one element")
            image = image[0]
        image_path = self.image_dir / f"{prompt_index}.png"
        if save:
            image.save(image_path)
        return str(image_path)
    
    
    
    def _save_results(self):
        """Save experiment results to CSV and JSON"""
        results_df = pd.DataFrame(self.results)
        self._save_csv(results_df, self.exp_dir / "results.csv")
        
        metadata = {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": str(self.dataset.csv_path),  # Convert Path to string
            "num_prompts": len(self.dataset),
            # "attacker_config": self.get_serializable_config(self.attacker),
            # "model_config": self.get_serializable_config(self.model)
        }
        
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


    def _save_csv(self, data, save_dir: str):
        if not os.path.exists(save_dir):
            data.to_csv(save_dir, index=False)
        else:
            data.to_csv(save_dir, mode='a', header=False, index=False)
        
        
    def _process_attack_result(
        self,
        result,
        prompt_index: str
    ) -> Dict:
        """Process attack result and save artifacts"""
        processed = {
            "prompt_index": prompt_index,
            "original_prompt": result.original_prompt,
            "attack_prompt": result.attack_prompt,
            "success": result.success,
            "execution_time": result.execution_time,
            "is_text_NSFW": result.is_text_NSFW,
            "is_image_NSFW": result.is_image_NSFW
        }
        
        # Save generated image if available
        if result.generated_image and self.config.save_images:
            image_path = self._save_image(
                result.generated_image,
                prompt_index,
                # len(self.results)
            )
            processed["image_path"] = image_path
            
        if result.metadata:
            processed.update(result.metadata)
            
        return processed
        
    def run(self):
            """Run the benchmark experiment using pre-processed attack prompts"""
            self.logger.info(f"Starting experiment: {self.config.experiment_name}")
            self.logger.info(f"Model: {self.config.model.name}")
            self.logger.info(f"Dataset: {self.dataset_path}")
            self.logger.info(f"Attack method: {self.config.attack_method}")
            self.logger.info(f"Processing {len(self.dataset)} prompts")
            
            try:
                for idx in tqdm(range(len(self.dataset))):
                    sample = self.dataset[idx]
                    prompt_index = sample['index']
                    original_prompt = sample['original_prompt']
                    attack_prompt = sample['attack_prompt']
                    prompt_image_save_path = self._save_image(original_prompt, prompt_index, save=False)

                    if os.path.exists(prompt_image_save_path):
                        self.logger.info(f"Image for prompt {prompt_index} already exists, skipping.")
                        self.logger.info(f"{prompt_image_save_path} | Original Prompt: {original_prompt}")
                        continue
                        
                    self.logger.debug(f"Processing prompt {prompt_index}")
                
                    try:
                        # Use both original_prompt and attack_prompt
                        self.logger.info(f"Original Prompt: {original_prompt} | Attack_prmopt: {attack_prompt}")
                        result = self.attacker(original_prompt, attack_prompt=attack_prompt)
                        processed_result = self._process_attack_result(result, prompt_index)
                        self.results.append(processed_result)
                    except Exception as e:
                        raise ValueError(f"Error processing prompt {prompt_index}: {str(e)}")
                        
                self._save_results()
                self.logger.info("Experiment completed successfully")
                
            except Exception as e:
                raise ValueError(f"Experiment failed: {str(e)}")        
    
    
    def get_summary(self) -> Dict:
        """Return summary statistics of the experiment"""
        if not self.results:
            return {}
            
        results_df = pd.DataFrame(self.results)
        
        summary = {
            "model": self.config.model.name,
            "dataset": Path(self.dataset_path).stem,
            "attack_method": self.config.attack_method,
            "total_prompts": len(results_df),
            "successful_attacks": results_df["success"].sum(),
            "success_rate": results_df["success"].mean(),
            "avg_execution_time": results_df["execution_time"].mean(),
            "bypass_detector_rate": results_df["is_text_NSFW"].mean(),
            "bypass_detector_rate": results_df["is_image_NSFW"].mean()
        }
        
        return summary



def run_benchmark(config_path: str):
    """Run benchmark experiments based on YAML config"""
    config = BenchmarkConfig.from_yaml(config_path)
    
    all_summaries = []
    
    for dataset_path in config.datasets:
        try:
            experiment = BenchmarkExperiment(
                config=config,
                dataset_path=dataset_path
            )
            experiment.run()
            summary = experiment.get_summary()
            all_summaries.append(summary)
        except Exception as e:
            print(f"Error running experiment with dataset={dataset_path}: {str(e)}")
            raise
    
    summary_df = pd.DataFrame(all_summaries)
    summary_path = Path(config.output_dir) / \
                config.attack_method / \
                config.model.name / \
                f"{config.attack_method}_{config.model.name}_summary.csv"
    
    if not summary_path.exists():
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)

    print(f"Overall summary saved to {summary_path}")
    
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark experiments')
    parser.add_argument('--config_path', 
                        type=str, 
                        default='config/stable-diffusion-3.5-medium_noattack.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    
    run_benchmark(args.config_path)