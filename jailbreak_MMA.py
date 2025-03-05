import os
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jailbreak_diffusion.diffusion_model import DiffusionFactory

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    attack_method: str
    params: Dict[str, Any]

@dataclass
class InferenceConfig:
    """Configuration for inference"""
    experiment_name: str
    output_dir: str
    log_level: str
    model: ModelConfig
    results_path: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InferenceConfig':
        """Create config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Convert model dict to ModelConfig
        model_dict = config_dict.pop('model')
        
        # Ensure all required fields are present
        if 'attack_method' not in model_dict:
            raise ValueError("Configuration error: 'attack_method' is required in model configuration")
            
        model_config = ModelConfig(**model_dict)
        
        return cls(model=model_config, **config_dict)

class InferenceRunner:
    def __init__(self, config: InferenceConfig):
        self.config = config
        
        # Setup model
        self.model = DiffusionFactory(
            **self.config.model.params
        )
        
        # Create output directory with model name
        # Format output directory with model name if placeholder exists
        output_dir_str = self.config.output_dir
        if "{model_name}" in output_dir_str:
            output_dir_str = output_dir_str.format(model_name=self.config.model.name)
        
        self.output_dir = Path(output_dir_str)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Load results
        self.results_data = self._load_results(self.config.results_path)
        
    def _setup_logging(self):
        """Configure logging"""
        logger_name = f"{self.config.experiment_name}_{self.config.model.name}"
        self.logger = logging.getLogger(logger_name)
        
        # Clear any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        self.logger.setLevel(self.config.log_level)
        
        # File handler
        log_file = self.output_dir / f"{logger_name}.log"
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
        
        # Avoid propagation to root logger
        self.logger.propagate = False
        
        self.logger.info(f"Logging file: {log_file}")
    
    def _load_results(self, results_path: str) -> Dict:
        """Load results JSON file and copy it to output directory"""
        self.logger.info(f"Loading results from {results_path}")
        
        # Load JSON data
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Copy the results file to the output directory for reference
        import shutil
        results_filename = Path(results_path).name
        output_results_path = self.output_dir / results_filename
        
        # Only copy if the file doesn't already exist
        if not output_results_path.exists():
            try:
                shutil.copy2(results_path, output_results_path)
                self.logger.info(f"Copied results file to {output_results_path}")
            except Exception as e:
                self.logger.warning(f"Failed to copy results file: {e}")
        else:
            self.logger.info(f"Results file already exists at {output_results_path}")
            
        self.logger.info(f"Loaded {len(data.get('prompts', []))} prompts")
        return data
    
    def run_inference(self):
        """Generate images from attack prompts in the results file"""
        prompts = self.results_data.get('prompts', [])
        total_prompts = len(prompts)
        self.logger.info(f"Starting inference for {total_prompts} prompts")
        
        # Process each prompt one by one
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
            prompt_id = prompt.get('id')
            attack_prompt = prompt.get('attack_prompt')
            
            if not attack_prompt:
                self.logger.warning(f"No attack prompt for ID {prompt_id}, skipping")
                continue
            
            output_path = self.output_dir / f"{prompt_id}.jpg"
            
            # Skip if image already exists
            if output_path.exists():
                # Use debug level to avoid cluttering the log with already processed images
                self.logger.debug(f"Image {output_path} already exists, skipping")
                continue
            
            try:
                self.logger.info(f"[{idx+1}/{total_prompts}] Generating image for prompt ID {prompt_id}")
                # Generate image using the model
                image = self.model.generate(attack_prompt)
                
                # Save the image
                if isinstance(image, list) and len(image) > 0:
                    image = image[0].images  # Get first image if returned as list
                
                # Save as JPG
                if image:
                    image.images[0].save(output_path)
                    self.logger.info(f"Saved image to {output_path}")
                else:
                    self.logger.error(f"Failed to generate image for prompt ID {prompt_id}")
            
            except Exception as e:
                self.logger.error(f"Error generating image for prompt ID {prompt_id}: {str(e)}")
                # Continue with next prompt even if this one fails
                continue

def main(config_path: str):
    """Run inference based on YAML config"""
    config = InferenceConfig.from_yaml(config_path)
    
    runner = InferenceRunner(config)
    runner.run_inference()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run inference from attack prompts')
    parser.add_argument('--config_path', 
                        type=str, 
                        required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()
    
    main(args.config_path)