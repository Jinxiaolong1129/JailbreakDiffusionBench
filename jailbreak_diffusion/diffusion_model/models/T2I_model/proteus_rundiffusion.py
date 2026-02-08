# jailbreak_diffusion/diffusion_model/models/T2I_model/proteus_rundiffusion.py
from diffusers import DiffusionPipeline
import torch
from typing import Optional, Dict, Any
from .base import BaseDiffusionModel
from ...core.outputs import GenerationInput, GenerationOutput
import time

class ProteusModel:
    def __init__(
        self,
        model_name: str = "dataautogpt3/Proteus-RunDiffusion",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,  # Changed to float16 as it's more commonly supported
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = self.load_model()
        
    def load_model(self):
        """Load model using DiffusionPipeline"""
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="balanced"
        )
        # .to(self.device)
        
        if hasattr(pipeline, 'vae'):
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
        
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        params = input_data.extra_params or {}
        all_images = []
        start_time = time.time()
        
        generation_params = {
            "prompt": None,
            "guidance_scale": params.get("guidance_scale", 7.5),  # Adjusted default value
            "num_images_per_prompt": params.get("num_images_per_prompt", 1),
            "num_inference_steps": params.get("num_inference_steps", 30),  # Reduced default steps
            "width": params.get("width", 768),  # Changed default size
            "height": params.get("height", 768),  # Changed default size
        }
        
        for prompt in input_data.prompts:
            if not prompt.startswith("score_"):
                prompt = f"score_9, {prompt}"
                
            generation_params["prompt"] = prompt
            print(f"Generating with params: {generation_params}")
            
            try:
                output = self.model(**generation_params)
                all_images.extend(output.images)
            except Exception as e:
                print(f"Error generating image for prompt '{prompt}': {str(e)}")
                continue
            
        generation_time = time.time() - start_time
                
        return GenerationOutput(
            images=all_images,
            metadata={
                "model": self.model_name,
                "parameters": params,
                "generation_time": generation_time
            }
        )