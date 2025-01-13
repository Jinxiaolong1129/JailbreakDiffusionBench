import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import json
from tqdm import tqdm
from jailbreak_diffusion.attack import AttackerFactory
from jailbreak_diffusion.diffusion_model.models import DiffusionFactory


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments"""
    experiment_name: str
    output_dir: str = "benchmark_results"
    save_images: bool = True
    save_prompts: bool = True
    log_level: str = "INFO"
    batch_size: int = 1
    num_workers: int = 1

class PromptDataset:
    """Dataset class for loading and managing prompts"""
    
    def __init__(self, csv_path: str):
        """
        Initialize dataset from CSV file
        
        Args:
            csv_path: Path to CSV file containing prompts
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self._validate_data()
        
    def _validate_data(self):
        """Validate that CSV has required columns"""
        required_cols = ['prompt_id', 'prompt']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'prompt_id': row['prompt_id'],
            'prompt': row['prompt']
        }

class BenchmarkExperiment:
    """Main experiment class for running jailbreak benchmarks"""
    
    def __init__(
        self,
        config: BenchmarkConfig,
        attacker: Any,
        model: Any,
        dataset: PromptDataset
    ):
        """
        Initialize benchmark experiment
        
        Args:
            config: Experiment configuration
            attacker: Attacker instance to use
            model: Target model to attack
            dataset: Dataset containing prompts
        """
        self.config = config
        self.attacker = attacker
        self.model = model
        self.dataset = dataset
        
        # Set up logging
        self._setup_logging()
        
        # Create output directories
        self.exp_dir = self._setup_experiment_dir()
        self.image_dir = self.exp_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Initialize results tracking
        self.results = []
        
    def _setup_logging(self):
        """Configure logging"""
        self.logger = logging.getLogger(self.config.experiment_name)
        self.logger.setLevel(self.config.log_level)
        
        # File handler
        fh = logging.FileHandler(
            f"{self.config.output_dir}/{self.config.experiment_name}.log"
        )
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
        
    def _setup_experiment_dir(self) -> Path:
        """Create and return experiment directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(self.config.output_dir) / f"{self.config.experiment_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
        
    def _save_image(self, image: Any, prompt_id: str, idx: int) -> str:
        """Save generated image and return path"""
        image_path = self.image_dir / f"{prompt_id}_{idx}.png"
        image.save(image_path)
        return str(image_path)
        
    def _save_results(self):
        """Save experiment results to CSV and JSON"""
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.exp_dir / "results.csv", index=False)
        
        # Save experiment metadata
        metadata = {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset.csv_path,
            "num_prompts": len(self.dataset),
            "attacker_config": self.attacker.__dict__,
            "model_config": self.model.__dict__
        }
        
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def _process_attack_result(
        self,
        result,
        prompt_id: str
    ) -> Dict:
        """Process attack result and save artifacts"""
        processed = {
            "prompt_id": prompt_id,
            "original_prompt": result.original_prompt,
            "attack_prompt": result.attack_prompt,
            "success": result.success,
            "execution_time": result.execution_time,
            "bypass_detector": result.bypass_detector,
            "bypass_checker": result.bypass_checker
        }
        
        # Save generated image if available
        if result.generated_image and self.config.save_images:
            image_path = self._save_image(
                result.generated_image,
                prompt_id,
                len(self.results)
            )
            processed["image_path"] = image_path
            
        # Add any additional metadata
        if result.metadata:
            processed.update(result.metadata)
            
        return processed
        
    def run(self):
        """Run the benchmark experiment"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Processing {len(self.dataset)} prompts")
        
        try:
            for idx in tqdm(range(len(self.dataset))):
                sample = self.dataset[idx]
                prompt_id = sample['prompt_id']
                prompt = sample['prompt']
                
                self.logger.debug(f"Processing prompt {prompt_id}")
                
                # Run attack
                try:
                    result = self.attacker.attack(prompt)
                    processed_result = self._process_attack_result(result, prompt_id)
                    self.results.append(processed_result)
                except Exception as e:
                    self.logger.error(f"Error processing prompt {prompt_id}: {str(e)}")
                    continue
                    
            # Save final results
            self._save_results()
            self.logger.info("Experiment completed successfully")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
            
    def get_summary(self) -> Dict:
        """Return summary statistics of the experiment"""
        if not self.results:
            return {}
            
        results_df = pd.DataFrame(self.results)
        
        summary = {
            "total_prompts": len(results_df),
            "successful_attacks": results_df["success"].sum(),
            "success_rate": results_df["success"].mean(),
            "avg_execution_time": results_df["execution_time"].mean(),
            "bypass_detector_rate": results_df["bypass_detector"].mean(),
            "bypass_checker_rate": results_df["bypass_checker"].mean()
        }
        
        return summary

    
    
if __name__ == "__main__":
    # Create configuration
    config = BenchmarkConfig(
        experiment_name="jailbreak_test_1",
        output_dir="benchmark_results",
        save_images=True
    )

    # Create dataset
    dataset = PromptDataset("/home/ubuntu/xiaolong/jailbreakbench/data/I2P/04-Sex.csv")

    model = DiffusionFactory("stable-diffusion-3-medium")
    attacker = AttackerFactory("PGJ", target_model=model)

    experiment = BenchmarkExperiment(config, attacker, model, dataset)
    experiment.run()

    summary = experiment.get_summary()
    print(summary)