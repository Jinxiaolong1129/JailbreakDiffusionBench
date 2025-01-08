
#  diffusion_output.py
from enum import Enum, auto
from typing import Optional, Union, List
from dataclasses import dataclass
from PIL import Image
import torch

class ModelType(str, Enum):
    TEXT_TO_IMAGE = "text2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINTING = "inpainting"

class ModelArchitecture(str, Enum):
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    KANDINSKY = "kandinsky"
    IF = "if"  # DeepFloyd IF

@dataclass
class GenerationInput:
    prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    input_images: Optional[List[Image.Image]] = None
    image_strength: Optional[float] = 0.8  # For img2img
    mask_images: Optional[List[Image.Image]] = None  # For inpainting

@dataclass
class GenerationOutput:
    images: List[Image.Image]
    prompt_tokens: List[int]
    latents: Optional[List[torch.Tensor]] = None
    metadata: dict = None
