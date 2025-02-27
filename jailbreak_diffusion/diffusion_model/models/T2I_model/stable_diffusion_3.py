# jailbreak_diffusion/diffusion_model/models/T2I_model/stable_diffusion.py

from diffusers import (
    StableDiffusion3Pipeline,
)
import torch
from typing import Optional, Dict, Any
from ...core.outputs import GenerationInput, GenerationOutput
import time

class StableDiffusion3Model:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        if model_name ==  "stabilityai/stable-diffusion-3-medium-diffusers":
            self.torch_dtype = torch.float16
        elif model_name == "stabilityai/stable-diffusion-3.5-medium":
            self.torch_dtype = torch.float16
        elif model_name == "stabilityai/stable-diffusion-3-turbo":
            self.torch_dtype = torch.float16
        elif "stable-diffusion-3.5-large-turbo" in model_name:
            self.torch_dtype = torch.float16
        elif "stabilityai/stable-diffusion-3.5-large" in model_name:
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch_dtype
        print(f"Loading model {self.model_name}")
        print(f"Device: {self.device}")
        print(f"torch_dtype: {self.torch_dtype}")
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load model using DiffusionPipeline"""
        pipeline = StableDiffusion3Pipeline.from_pretrained(
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
        
        # Set default parameters based on model type (turbo vs regular)
        is_turbo = "turbo" in self.model_name
        
        generation_params = {
            "prompt": None,  
            "negative_prompt": input_data.negative_prompt[0] if input_data.negative_prompt else None,
            "num_inference_steps": params.get("num_inference_steps", 4 if is_turbo else 28),
            "guidance_scale": params.get("guidance_scale", 0.0 if is_turbo else 4.5),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "max_sequence_length": params.get("max_sequence_length", 512),
        }
        
        for prompt in input_data.prompts:
            generation_params["prompt"] = prompt
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