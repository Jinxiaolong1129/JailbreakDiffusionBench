# jailbreak_diffusion/diffusion_model/models/T2I_model/cogview.py
from diffusers import CogView3PlusPipeline
import torch
from typing import Optional, Dict, Any
from .base import BaseDiffusionModel
from ...core.outputs import GenerationInput, GenerationOutput
import time

class CogViewModel:
    def __init__(
        self,
        model_name: str = "THUDM/CogView3-Plus-3B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = self.load_model()
        
    def load_model(self):
        """Load model using CogView3PlusPipeline"""
        pipeline = CogView3PlusPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="balanced"
        )
        # .to(self.device)
        # pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()
        
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        params = input_data.extra_params or {}
        all_images = []
        start_time = time.time()
        
        generation_params = {
            "prompt": None,
            "guidance_scale": params.get("guidance_scale", 7.0),
            "num_images_per_prompt": params.get("num_images_per_prompt", 1),
            "num_inference_steps": params.get("num_inference_steps", 50),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
        }
        
        for prompt in input_data.prompts:
            generation_params["prompt"] = prompt
            print(f"Generating with params: {generation_params}")
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