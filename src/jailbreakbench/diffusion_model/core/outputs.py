# diffusion_model/core/outputs.py

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from PIL import Image
import torch

@dataclass
class GenerationInput:
    """Unified input structure for all generation models"""
    prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    input_images: Optional[List[Image.Image]] = None
    mask_images: Optional[List[Image.Image]] = None
    image_strength: Optional[float] = 0.8
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    
    # Video-specific parameters
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    motion_bucket_id: Optional[int] = None
    
    # Additional parameters as dict
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert input to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class GenerationOutput:
    """Unified output structure for generation results"""
    images: Optional[List[Image.Image]] = None
    videos: Optional[List[torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None
    generation_time: Optional[float] = None
    
    def merge(self, other: 'GenerationOutput') -> 'GenerationOutput':
        """Merge another output into this one"""
        if other.images:
            self.images = (self.images or []) + other.images
        if other.videos:
            self.videos = (self.videos or []) + other.videos
        if other.metadata:
            self.metadata = {**(self.metadata or {}), **other.metadata}
        return self