# diffusion_model/core/wrapper.py

from abc import ABC, abstractmethod
from pathlib import Path
import json
import datetime
import logging
from typing import List, Optional, Dict, Any
import torch
from PIL import Image

class DiffusionWrapper(ABC):
    """Base wrapper for all generation models"""
    
    def __init__(self, save_dir: str = "outputs", device: Optional[str] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs):
        """Generate outputs from prompts"""
        pass
    
    def save_outputs(self, output: 'GenerationOutput', prompts: List[str]):
        """Save generation outputs and metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images
        if output.images:
            for i, (image, prompt) in enumerate(zip(output.images, prompts)):
                # Save image
                image_path = self.save_dir / f"{timestamp}_{i}.png"
                image.save(image_path)
                
                # Save metadata
                metadata_path = self.save_dir / f"{timestamp}_{i}_metadata.json"
                metadata = {
                    "timestamp": timestamp,
                    "prompt": prompt,
                    "generation_time": output.generation_time,
                    **(output.metadata or {})
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Save videos if present
        if output.videos:
            # Add video saving logic here
            pass
            
        self.logger.info(f"Saved {len(output.images or [])} images to {self.save_dir}")
