from collections.abc import Iterator

from pydantic import BaseModel


class LLMOutput(BaseModel):
    responses: list[str]
    prompt_tokens: list[int]
    completion_tokens: list[int]

    def __getitem__(self, item) -> tuple[str, int, int]:
        return self.responses[item], self.prompt_tokens[item], self.completion_tokens[item]

    def iterate_elements(self) -> Iterator[tuple[str, int, int]]:
        for i in range(len(self.responses)):
            yield self[i]

    @classmethod
    def from_list(cls, inputs: list[tuple[str, int, int]]) -> "LLMOutput":
        return cls(
            responses=[i[0] for i in inputs],
            prompt_tokens=[i[1] for i in inputs],
            completion_tokens=[i[2] for i in inputs],
        )




# model_types.py
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
