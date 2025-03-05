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
import copy
from jailbreak_diffusion.attack import AttackerFactory
from jailbreak_diffusion.diffusion_model import DiffusionFactory
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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
    """Dataset class for loading and managing prompts from JSON file"""
    
    def __init__(self, json_path: str, start_id: int = None, end_id: int = None):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract prompts
        self.prompts = data.get("prompts", [])
        
        # Filter by ID range if specified
        if start_id is not None or end_id is not None:
            start_id = start_id if start_id is not None else 1
            end_id = end_id if end_id is not None else float('inf')
            
            self.prompts = [p for p in self.prompts if start_id <= p.get("id", 0) <= end_id]
            
        self._validate_data()
        
    def _validate_data(self):
        if not self.prompts:
            raise ValueError(f"No prompts found in dataset or after ID filtering")
        
        required_fields = ['id', 'text']
        for prompt in self.prompts:
            missing = [field for field in required_fields if field not in prompt]
            if missing:
                raise ValueError(f"Prompt missing required fields: {missing}")
            
    def __len__(self):
        return len(self.prompts)
        
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {
            'index': str(prompt['id']),
            'prompt': prompt['text']
        }

class BenchmarkExperiment:
    def __init__(
        self,
        config: BenchmarkConfig,
        dataset_path: str,
        start_id: int = None,
        end_id: int = None
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
        
        # Setup dataset with ID filtering
        self.dataset = PromptDataset(dataset_path, start_id, end_id)
        
        # Create base output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create experiment directory with new structure
        self.exp_dir = self._setup_experiment_dir()
        self.image_dir = self.exp_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self.logger.info(f"Processing prompts with IDs from {start_id} to {end_id}")
        
        self.results = []
        
    def _setup_experiment_dir(self) -> Path:
        """Create and return experiment directory with new structure"""
        # Directory structure: benchmark_results/[attack_method]/[model_name]/[dataset_name]
        exp_dir = Path(self.config.output_dir) / \
                 self.config.attack_method / \
                 self.config.model.name / \
                 Path(self.dataset_path).stem
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _setup_logging(self):
        """Configure logging"""
        exp_name = f"{self.config.attack_method}_{self.config.model.name}_{Path(self.dataset_path).stem}"
        
        # Get logger and clear any existing handlers
        self.logger = logging.getLogger(exp_name)
        
        # Remove all handlers associated with the logger object
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                
        # Also clear root logger handlers to prevent duplication    
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
        self.logger.setLevel(self.config.log_level)
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
        
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
        if save:
            image.save(image_path)
        return str(image_path)
    
    def _save_results(self):
        """Save experiment results to JSON and create enriched dataset"""
        # Save traditional results
        results_path = self.exp_dir / "results.json"
        
        # Create results object with metadata
        results_obj = {
            "metadata": {
                "experiment_name": self.config.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "dataset": str(self.dataset.json_path),
                "num_prompts": len(self.dataset),
            },
            "results": self.results
        }
        
        # Check if file exists to handle appending
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Append new results
                existing_data["results"].extend(self.results)
                # Update metadata
                existing_data["metadata"]["timestamp"] = results_obj["metadata"]["timestamp"]
                existing_data["metadata"]["num_prompts"] = len(existing_data["results"])
                
                results_obj = existing_data
            except json.JSONDecodeError:
                self.logger.warning(f"Could not read existing results.json. Creating new file.")
        
        # Write the results
        with open(results_path, 'w') as f:
            json.dump(results_obj, f, indent=2, cls=NumpyEncoder)
            
        # Create enriched dataset by combining original JSON with results
        self._create_enriched_dataset()

    def _create_enriched_dataset(self):
        """Create enriched dataset by combining original JSON with results"""
        # Create a mapping of prompt_id to result for quick lookup
        result_map = {result["prompt_index"]: result for result in self.results}
        
        # Load original dataset JSON
        with open(self.dataset.json_path, 'r') as f:
            original_data = json.load(f)
        
        # Create a copy of the original dataset
        enriched_data = copy.deepcopy(original_data)
        
        # Enrich each prompt with attack results if available
        for prompt in enriched_data.get("prompts", []):
            prompt_id = str(prompt.get("id", ""))
            if prompt_id in result_map:
                result = result_map[prompt_id]
                # Add attack results to the prompt
                prompt["attack_prompt"] = result.get("attack_prompt", "")
                prompt["image_path"] = result.get("image_path", "")
                prompt["success"] = result.get("success", False)
                prompt["execution_time"] = result.get("execution_time", 0)
                prompt["is_text_NSFW"] = result.get("is_text_NSFW", False)
                prompt["is_image_NSFW"] = result.get("is_image_NSFW", False)
                
                # Add any additional metadata
                if "metadata" in result:
                    prompt["attack_metadata"] = result["metadata"]
        
        # Add experiment metadata
        enriched_data["experiment_metadata"] = {
            "experiment_name": self.config.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model.name,
            "attack_method": self.config.attack_method
        }
        
        # Save enriched dataset
        enriched_path = self.exp_dir / "enriched_dataset.json"
        with open(enriched_path, 'w') as f:
            json.dump(enriched_data, f, indent=2, cls=NumpyEncoder)
            
        self.logger.info(f"Enriched dataset saved to {enriched_path}")
        
    def _process_attack_result(
        self,
        result,
        prompt_index: str
    ) -> Dict:
        """Process attack result and save artifacts"""
        processed = {
            "prompt_index": prompt_index,
            "original_prompt": result.original_prompt,
            "attack_prompt": result.attack_prompt[0] if isinstance(result.attack_prompt, list) else result.attack_prompt,
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
            )
            processed["image_path"] = image_path
        else:
            processed["image_path"] = self._save_image(None, prompt_index, save=False)    
            
        if result.metadata:
            processed["metadata"] = result.metadata
            
        return processed
        
    def run(self):
        """Run the benchmark experiment"""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Model: {self.config.model.name}")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Attack method: {self.config.attack_method}")
        self.logger.info(f"Processing {len(self.dataset)} prompts")
        
        try:
            for idx in tqdm(range(len(self.dataset))):
                sample = self.dataset[idx]
                prompt_index = sample['index']
                prompt = sample['prompt']
                
                # Check if the image for this prompt already exists
                prompt_image_save_path = self._save_image(None, prompt_index, save=False)
                if os.path.exists(prompt_image_save_path):
                    self.logger.info(f"Image for prompt {prompt_index} already exists, skipping.")
                    self.logger.info(f"{prompt_image_save_path} | Prompt: {prompt}")
                    continue
                    
                self.logger.debug(f"Processing prompt {prompt_index}")
            
                try:
                    result = self.attacker(prompt)
                    processed_result = self._process_attack_result(result, prompt_index)
                    self.results.append(processed_result)
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    self.logger.error(f"Error processing prompt {prompt_index}: {str(e)}\n{error_trace}")
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
            "model": self.config.model.name,
            "dataset": Path(self.dataset_path).stem,
            "attack_method": self.config.attack_method,
            "total_prompts": len(results_df),
            "successful_attacks": results_df["success"].sum() if "success" in results_df else 0,
            "success_rate": results_df["success"].mean() if "success" in results_df else 0,
            "avg_execution_time": results_df["execution_time"].mean() if "execution_time" in results_df else 0,
            "text_NSFW_rate": results_df["is_text_NSFW"].mean() if "is_text_NSFW" in results_df else 0,
            "image_NSFW_rate": results_df["is_image_NSFW"].mean() if "is_image_NSFW" in results_df else 0
        }
        
        return summary

def run_benchmark(config_path: str, start_id: int = None, end_id: int = None):
    """Run benchmark experiments based on YAML config with ID filtering"""
    config = BenchmarkConfig.from_yaml(config_path)
    
    all_summaries = []
    
    for dataset_path in config.datasets:
        try:
            experiment = BenchmarkExperiment(
                config=config,
                dataset_path=dataset_path,
                start_id=start_id,
                end_id=end_id
            )
            experiment.run()
            summary = experiment.get_summary()
            all_summaries.append(summary)
        except Exception as e:
            print(f"Error running experiment with dataset={dataset_path}: {str(e)}")
            raise
    
    # Only create summary if there are results
    if all_summaries:
        # Add ID range information to filename
        id_suffix = ""
        if start_id is not None or end_id is not None:
            start_str = str(start_id) if start_id is not None else "start"
            end_str = str(end_id) if end_id is not None else "end"
            id_suffix = f"_id{start_str}-{end_str}"
        
        summary_path = Path(config.output_dir) / \
                    config.attack_method / \
                    config.model.name / \
                    f"{config.attack_method}_{config.model.name}{id_suffix}_summary.json"
        
        summary_obj = {
            "timestamp": datetime.now().isoformat(),
            "id_range": {"start_id": start_id, "end_id": end_id},
            "summaries": all_summaries
        }
        
        if not summary_path.exists():
            # Create new summary file
            with open(summary_path, 'w') as f:
                json.dump(summary_obj, f, indent=2, cls=NumpyEncoder)
        else:
            try:
                # Append to existing summary
                with open(summary_path, 'r') as f:
                    existing_data = json.load(f)
                
                # If existing data has different structure, save to a new file
                if "summaries" not in existing_data:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    new_path = summary_path.with_name(f"{summary_path.stem}_{timestamp}.json")
                    with open(new_path, 'w') as f:
                        json.dump(summary_obj, f, indent=2, cls=NumpyEncoder)
                    print(f"Structure mismatch with existing summary. Saved to {new_path}")
                else:
                    # Append summaries and update timestamp
                    existing_data["summaries"].extend(summary_obj["summaries"])
                    existing_data["timestamp"] = summary_obj["timestamp"]
                    
                    with open(summary_path, 'w') as f:
                        json.dump(existing_data, f, indent=2, cls=NumpyEncoder)
            
            except json.JSONDecodeError:
                # If JSON is invalid, create a new file with timestamp
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                new_path = summary_path.with_name(f"{summary_path.stem}_{timestamp}.json")
                with open(new_path, 'w') as f:
                    json.dump(summary_obj, f, indent=2, cls=NumpyEncoder)
                print(f"Error reading existing summary. Saved to {new_path}")
        
        print(f"Overall summary saved to {summary_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark experiments')
    parser.add_argument('--config_path', 
                        type=str, 
                        default='config/stable-diffusion-3.5-medium_noattack.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--start_id', 
                        type=int, 
                        default=None,
                        help='Starting ID to process (inclusive)')
    parser.add_argument('--end_id', 
                        type=int, 
                        default=None,
                        help='Ending ID to process (inclusive)')
    args = parser.parse_args()
    
    run_benchmark(args.config_path, args.start_id, args.end_id)
    
    
    
    