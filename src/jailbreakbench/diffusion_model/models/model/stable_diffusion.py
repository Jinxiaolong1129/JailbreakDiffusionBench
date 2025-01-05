# diffusion_model/models/model/stable_diffusion.py

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
import torch
from typing import Optional, Dict, Any
from .base import BaseDiffusionModel
from core.outputs import GenerationInput, GenerationOutput
import time


class StableDiffusionModel(BaseDiffusionModel):
    """Implementation for Stable Diffusion models"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        is_xl: bool = False,
        **kwargs
    ):
        super().__init__(model_path, device, torch_dtype)
        self.is_xl = is_xl
        self.kwargs = kwargs
        self.safety_checker = kwargs.get("safety_checker", None)
        
    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            safety_checker=self.safety_checker,
            use_safetensors=True
        ).to(self.device)
        
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        all_images = []
        start_time = time.time()
        
        for prompt, negative_prompt in zip(
            input_data.prompts,
            input_data.negative_prompts or [None] * len(input_data.prompts)
        ):
            output = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=input_data.num_inference_steps,
                guidance_scale=input_data.guidance_scale,
                width=input_data.extra_params.get("width", 512),
                height=input_data.extra_params.get("height", 512),
            )
            all_images.extend(output.images)
            
        generation_time = time.time() - start_time
                
        return GenerationOutput(
            images=all_images,
            metadata={
                "model": self.model_path,
                "parameters": input_data.to_dict(),
                "generation_time": generation_time
            }
        )

class StableDiffusionXLModel(StableDiffusionModel):
    """Implementation for SDXL models with optional refiner"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        use_refiner: bool = False,
        **kwargs
    ):
        super().__init__(model_path, device, torch_dtype, is_xl=True, **kwargs)
        self.use_refiner = use_refiner
        self.refiner = None
        if use_refiner:
            self.refiner_steps = kwargs.get("refiner_steps", 20)
            
    def load_model(self):
        base = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True
        ).to(self.device)
        
        if self.use_refiner:
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            ).to(self.device)
            
        return base
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        all_images = []
        start_time = time.time()
        
        for prompt, negative_prompt in zip(
            input_data.prompts,
            input_data.negative_prompts or [None] * len(input_data.prompts)
        ):
            # Base model inference
            output = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=input_data.num_inference_steps,
                guidance_scale=input_data.guidance_scale,
                width=input_data.extra_params.get("width", 1024),
                height=input_data.extra_params.get("height", 1024),
            )
            
            # Refiner if enabled
            if self.use_refiner and self.refiner:
                output = self.refiner(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=output.images,
                    num_inference_steps=self.refiner_steps,
                ).images
                
            all_images.extend(output)
            
        generation_time = time.time() - start_time
                
        return GenerationOutput(
            images=all_images,
            metadata={
                "model": self.model_path,
                "parameters": input_data.to_dict(),
                "generation_time": generation_time,
                "refiner_used": self.use_refiner
            }
        )