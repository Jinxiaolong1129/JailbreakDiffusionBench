# exp_no_attack.py
import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
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

class JSONPromptDataset(Dataset):
    """Dataset class for loading and managing prompts from JSON files"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json(json_path)
        self.prompts = self.data.get('prompts', [])
        self._validate_data()
        
    def _load_json(self, json_path: str) -> Dict:
        """Load JSON data from file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _validate_data(self):
        """Validate that the JSON data has the required structure"""
        if not isinstance(self.data, dict):
            raise ValueError(f"JSON data must be a dictionary, got {type(self.data)}")
        
        if 'prompts' not in self.data:
            raise ValueError("JSON data missing 'prompts' key")
            
        if not isinstance(self.prompts, list):
            raise ValueError(f"'prompts' must be a list, got {type(self.prompts)}")
            
        if not self.prompts:
            raise ValueError("No prompts found in the dataset")
            
        # Check that each prompt has id and text fields
        for i, prompt in enumerate(self.prompts):
            if not isinstance(prompt, dict):
                raise ValueError(f"Prompt at index {i} must be a dictionary")
                
            if 'id' not in prompt:
                raise ValueError(f"Prompt at index {i} missing 'id' field")
                
            if 'text' not in prompt:
                raise ValueError(f"Prompt at index {i} missing 'text' field")
    
    def get_metadata(self) -> Dict:
        """Return dataset metadata"""
        return {
            "name": self.data.get("name", Path(self.json_path).stem),
            "description": self.data.get("description", ""),
            "size": len(self.prompts),
            "path": self.json_path,
            "category": Path(self.json_path).parent.parent.name,
            "source": Path(self.json_path).parent.name
        }
        
    def get_raw_data(self) -> Dict:
        """Return the raw dataset"""
        return self.data
            
    def __len__(self):
        return len(self.prompts)
        
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {
            'index': str(prompt['id']),  # Convert id to string for consistency
            'prompt': prompt['text'],
            'original_data': prompt  # Include original prompt data
        }

    def collate_fn(self, batch):
        """Custom collate function for batching"""
        indices = [item['index'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        original_data = [item['original_data'] for item in batch]
        return {
            'indices': indices,
            'prompts': prompts,
            'original_data': original_data
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
        
        # Setup dataset based on file extension
        if dataset_path.endswith('.json'):
            self.dataset = JSONPromptDataset(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format for {dataset_path}. Only JSON files are supported.")
        
        # Create base output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create experiment directory with new structure
        self.exp_dir = self._setup_experiment_dir()
        self.image_dir = self.exp_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        
        self.results = []
        
    def _setup_experiment_dir(self) -> Path:
        """Create and return experiment directory with new structure"""
        # Directory structure: benchmark_results/[attack_method]/[model_name]/[dataset_name]
        dataset_name = Path(self.dataset_path).stem
        dataset_category = Path(self.dataset_path).parent.parent.name  # benign or malicious
        dataset_source = Path(self.dataset_path).parent.name  # diffusion_db, coco, I2P, etc.
        
        exp_dir = Path(self.config.output_dir) / \
                 self.config.attack_method / \
                 self.config.model.name / \
                 dataset_category / dataset_source
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _setup_logging(self):
        """Configure logging"""
        dataset_name = Path(self.dataset_path).stem
        dataset_category = Path(self.dataset_path).parent.parent.name  # benign or malicious
        dataset_source = Path(self.dataset_path).parent.name  # diffusion_db, coco, I2P, etc.
        
        exp_name = f"{self.config.attack_method}_{self.config.model.name}_{dataset_category}_{dataset_source}"
        
        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(self.config.log_level)
        
        # Check if handlers already exist
        if not self.logger.handlers:
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

    def _save_image(self, image: Any, prompt_index: str, save=True) -> str:
        """Save generated image and return path"""
        if isinstance(image, list):
            if len(image) > 1:
                raise ValueError("Image list contains more than one element")
            image = image[0]
        image_path = self.image_dir / f"{prompt_index}.png"
        if save and image is not None:
            image.save(image_path)
        return str(image_path)
    
    def _save_results(self):
        """Save experiment results to CSV and JSON"""
        results_df = pd.DataFrame(self.results)
        results_path = self.exp_dir / "results.csv"
        self._save_csv(results_df, results_path)
        
        # Save original dataset
        dataset_json_path = self.exp_dir / "original_dataset.json"
        with open(dataset_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.dataset.get_raw_data(), f, indent=2)
        
        # Get dataset metadata
        dataset_metadata = self.dataset.get_metadata()
        
        metadata = {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_metadata,
            "model": {
                "name": self.config.model.name,
                "params": self.config.model.params
            },
            "attack_method": self.config.attack_method,
            "num_prompts_processed": len(self.results),
            "total_prompts_in_dataset": len(self.dataset)
        }
        
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.logger.info(f"Results saved to {results_path}")
        self.logger.info(f"Original dataset saved to {dataset_json_path}")


    def _save_csv(self, data, save_path: str):
        if not os.path.exists(save_path):
            data.to_csv(save_path, index=False)
        else:
            # If file exists, append without headers
            data.to_csv(save_path, mode='a', header=False, index=False)
        
    def _process_attack_results(
        self,
        results,
        prompt_indices: List[str],
        original_data: List[Dict]
    ) -> List[Dict]:
        """Process attack results for a batch and save artifacts"""
        processed_results = []
        
        for i, result in enumerate(results):
            prompt_index = prompt_indices[i]
            original = original_data[i] if i < len(original_data) else {}
            
            processed = {
                "prompt_index": prompt_index,
                "original_prompt": result.original_prompt,
                "attack_prompt": result.attack_prompt,
                "success": result.success,
                "execution_time": result.execution_time,
                "is_text_NSFW": result.is_text_NSFW,
                "is_image_NSFW": result.is_image_NSFW,
                # Include original prompt data
                "original_data": original
            }
            
            # Save generated image if available
            if result.generated_image and self.config.save_images:
                image_path = self._save_image(
                    result.generated_image,
                    prompt_index,
                )
                processed["image_path"] = image_path
                
            if result.metadata:
                processed.update(result.metadata)
                
            processed_results.append(processed)
            
        return processed_results
    
    def _batch_attack(self, prompts: List[str]) -> Tuple[List, List[Exception]]:
        """Execute attacks in batch"""
        results = []
        errors = []
        
        # For now, we'll still process sequentially but batched
        # In the future, this could be parallelized if the attacker supports it
        for prompt in prompts:
            result = self.attacker(prompt)
            results.append(result)
            errors.append(None)

                
        return results, errors
    
    def run(self):
        """Run the benchmark experiment with batch processing"""
        dataset_metadata = self.dataset.get_metadata()
        
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Model: {self.config.model.name}")
        self.logger.info(f"Dataset: {dataset_metadata['category']}/{dataset_metadata['source']}/{dataset_metadata['name']}")
        self.logger.info(f"Attack method: {self.config.attack_method}")
        self.logger.info(f"Processing {len(self.dataset)} prompts with batch size {self.config.batch_size}")
        
        # Setup DataLoader with specified batch size and workers
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.dataset.collate_fn,
            shuffle=False
        )
        
        total_batches = len(data_loader)
        self.logger.info(f"Total batches: {total_batches}")
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
            indices = batch['indices']
            prompts = batch['prompts']
            original_data = batch['original_data']
            
            self.logger.debug(f"Processing batch {batch_idx+1}/{total_batches} with {len(prompts)} prompts")
            
            # Check if images already exist for all prompts in batch
            skip_batch = True
            for prompt_index in indices:
                prompt_image_save_path = self._save_image(None, prompt_index, save=False)
                if not os.path.exists(prompt_image_save_path):
                    skip_batch = False
                    break
            
            if skip_batch:
                self.logger.info(f"Images for all prompts in batch {batch_idx+1} already exist, skipping.")
                continue
            
            # Execute batch of attacks
            batch_results, batch_errors = self._batch_attack(prompts)
            
            # Log errors if any
            for i, error in enumerate(batch_errors):
                if error is not None:
                    prompt_index = indices[i] if i < len(indices) else "unknown"
                    self.logger.error(f"Error processing prompt {prompt_index}: {str(error)}")
            
            # Process successful results
            valid_results = [r for r in batch_results if r is not None]
            valid_indices = [indices[i] for i, r in enumerate(batch_results) if r is not None]
            valid_original_data = [original_data[i] for i, r in enumerate(batch_results) if r is not None]
            
            if valid_results:
                processed_results = self._process_attack_results(valid_results, valid_indices, valid_original_data)
                self.results.extend(processed_results)
            
            # Save intermediate results every 10 batches
            if (batch_idx + 1) % 10 == 0:
                self._save_results()
                self.logger.info(f"Saved intermediate results after batch {batch_idx+1}/{total_batches}")
                
        # Save final results
        self._save_results()
        self.logger.info("Experiment completed successfully")
        

            
    def get_summary(self) -> Dict:
        """Return summary statistics of the experiment"""
        if not self.results:
            return {}
            
        results_df = pd.DataFrame(self.results)
        dataset_metadata = self.dataset.get_metadata()
        
        summary = {
            "model": self.config.model.name,
            "dataset_name": dataset_metadata["name"],
            "dataset_category": dataset_metadata["category"],
            "dataset_source": dataset_metadata["source"],
            "attack_method": self.config.attack_method,
            "total_prompts_processed": len(results_df),
            "total_prompts_in_dataset": len(self.dataset),
            "successful_attacks": results_df["success"].sum() if "success" in results_df else 0,
            "success_rate": results_df["success"].mean() if "success" in results_df else 0,
            "avg_execution_time": results_df["execution_time"].mean() if "execution_time" in results_df else None,
            "text_nsfw_rate": results_df["is_text_NSFW"].mean() if "is_text_NSFW" in results_df else None,
            "image_nsfw_rate": results_df["is_image_NSFW"].mean() if "is_image_NSFW" in results_df else None
        }
        
        return summary

def run_benchmark(config_path: str):
    """Run benchmark experiments based on YAML config"""
    config = BenchmarkConfig.from_yaml(config_path)
    
    all_summaries = []
    
    for dataset_path in config.datasets:
        experiment = BenchmarkExperiment(
            config=config,
            dataset_path=dataset_path
        )
        experiment.run()
        summary = experiment.get_summary()
        all_summaries.append(summary)


    if all_summaries:
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
    else:
        print("No summaries generated. All experiments failed.")
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark experiments')
    parser.add_argument('--config_path', 
                        type=str, 
                        default='config/no_attack/stable-diffusion-3-medium_noattack.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()
    
    run_benchmark(args.config_path)