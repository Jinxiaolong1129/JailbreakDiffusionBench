# diffusion_model/models/T2I_model/pixart_alpha.py

from diffusers import PixArtAlphaPipeline
import torch
from typing import Optional, Dict, Any
from ...core.outputs import GenerationInput, GenerationOutput
import time

class PixArtAlphaModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        compile_transformer: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.compile_transformer = compile_transformer
        
        print(f"Loading PixArt Alpha model {self.model_name}")
        print(f"Device: {self.device}")
        print(f"torch_dtype: {self.torch_dtype}")
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load PixArtAlpha model"""
        pipeline = PixArtAlphaPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        )
        pipeline = pipeline.to(self.device)
        
        # Enable memory optimization if needed
        # For torch versions < 2.0
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            if torch.__version__ < "2.0":
                pipeline.enable_xformers_memory_efficient_attention()
        
        # For torch versions >= 2.0, use torch.compile for performance boost
        if torch.__version__ >= "2.0" and self.compile_transformer:
            pipeline.transformer = torch.compile(
                pipeline.transformer, 
                mode="reduce-overhead", 
                fullgraph=True
            )
            
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        params = input_data.extra_params or {}
        all_images = []
        start_time = time.time()
        
        generation_params = {
            "prompt": None,  
            "negative_prompt": input_data.negative_prompt[0] if input_data.negative_prompt else None,
            "num_inference_steps": params.get("num_inference_steps", 50),
            "guidance_scale": params.get("guidance_scale", 7.5),
            "width": params.get("width", 512),
            "height": params.get("height", 512),
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