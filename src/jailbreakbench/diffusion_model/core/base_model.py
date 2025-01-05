# diffusion_model/core/base_model.py
from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from .model_types import ModelType, ModelArchitecture
from .outputs import GenerationInput, GenerationOutput

class DiffusionModelBase(ABC):
    """Abstract base class for all diffusion models."""
    
    def __init__(
        self,
        model_type: ModelType,
        model_arch: ModelArchitecture,
        device: str = "cuda"
    ):
        self.model_type = model_type
        self.model_arch = model_arch
        self.device = device
        self.logger = logging.getLogger(f"{model_arch}_{model_type}")
        
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize pipeline."""
        pass
    
    @abstractmethod
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        """Generate output from input data."""
        pass
    
    @abstractmethod
    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        pass

    def validate_inputs(self, input_data: GenerationInput) -> None:
        """Validate input data based on model type."""
        if self.model_type == ModelType.IMAGE_TO_IMAGE and input_data.input_images is None:
            raise ValueError("Input images required for image-to-image generation")
        if self.model_type == ModelType.INPAINTING and (
            input_data.input_images is None or input_data.mask_images is None
        ):
            raise ValueError("Both input images and masks required for inpainting")