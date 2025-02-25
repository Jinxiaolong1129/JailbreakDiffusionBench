# jailbreak_diffusion/diffusion_model/models/T2I_model/stable_diffusion.py

from diffusers import (
    DiffusionPipeline,
)
import torch
from typing import Optional, Dict, Any
from .base import BaseDiffusionModel
from ...core.outputs import GenerationInput, GenerationOutput
import time

class StableDiffusionModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        print(f"Loading model {self.model_name}")
        print(f"Device: {self.device}")
        print(f"torch_dtype: {self.torch_dtype}")
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load model using DiffusionPipeline"""
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            device_map="balanced"
        )
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        params = input_data.extra_params or {}
        all_images = []
        start_time = time.time()
        
        generation_params = {
            "prompt": None,  
            "negative_prompt": input_data.negative_prompt,
            "num_inference_steps": params.get("num_inference_steps", 50),
            # "guidance_scale": params.get("guidance_scale", 7.5),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
        }
        
        for prompt in input_data.prompts:
            generation_params["prompt"] = prompt
            print(generation_params)
            output = self.model(**generation_params)
            all_images.extend(output.images)
            
        generation_time = time.time() - start_time
                
        return GenerationOutput(
            images=all_images,
            metadata={
                "model": self.model_name,
                "parameters": params,
                "generation_time": generation_time
            }
        )
