from diffusers import FluxPipeline
import torch
from typing import Optional, Dict, Any
from .base import BaseDiffusionModel
from ...core.outputs import GenerationInput, GenerationOutput
import time

class FluxModel:
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = self.load_model()
        
    def load_model(self):
        """Load model using FluxPipeline"""
        pipeline = FluxPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            # device_map="balanced"
        ).to('cuda')
        # pipeline.enable_model_cpu_offload()  # Can be configured based on available GPU
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        params = input_data.extra_params or {}
        all_images = []
        start_time = time.time()
        
        generation_params = {
            "prompt": None,
            "height": params.get("height", 1024),
            "width": params.get("width", 1024),
            "guidance_scale": params.get("guidance_scale", 3.5),
            "num_inference_steps": params.get("num_inference_steps", 50),
            "max_sequence_length": params.get("max_sequence_length", 512),
            "generator": torch.Generator("cpu").manual_seed(params.get("seed", 0))
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