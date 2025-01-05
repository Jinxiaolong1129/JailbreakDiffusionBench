# diffusion_model/models/model/base.py

from abc import ABC, abstractmethod
import torch
from typing import Optional, List
from core.outputs import GenerationInput, GenerationOutput

class BaseDiffusionModel(ABC):
    """Base class for diffusion models"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = self.load_model()
        
    @abstractmethod
    def load_model(self):
        """Load model weights"""
        pass
        
    @abstractmethod
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        """Generate images"""
        pass

    def prepare_prompt(self, prompt: str) -> dict:
        """Prepare prompt for generation"""
        return {"prompt": prompt}