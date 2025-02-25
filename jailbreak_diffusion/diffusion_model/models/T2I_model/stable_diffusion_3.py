# diffusion_model/models/T2I_model/stable_diffusion_3.py

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline, T5EncoderModel
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
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_4bit = use_4bit
        
        print(f"Loading Stable Diffusion 3 model {self.model_name}")
        print(f"Device: {self.device}")
        print(f"torch_dtype: {self.torch_dtype}")
        print(f"use_4bit: {self.use_4bit}")
        
        self.model = self.load_model()
        
    def load_model(self):
        """Load StableDiffusion3 model with quantization"""
        
        # Configure quantization if enabled
        if self.use_4bit:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype
            )
            
            # Load transformer with quantization
            transformer = SD3Transformer2DModel.from_pretrained(
                self.model_name,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=self.torch_dtype
            )
            
            # For turbo models, we need to load the T5 encoder separately
            if "turbo" in self.model_name:
                t5_encoder = T5EncoderModel.from_pretrained(
                    "diffusers/t5-nf4", 
                    torch_dtype=self.torch_dtype
                )
                
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    self.model_name,
                    transformer=transformer,
                    text_encoder_3=t5_encoder,
                    torch_dtype=self.torch_dtype
                )
            else:
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    self.model_name,
                    transformer=transformer,
                    torch_dtype=self.torch_dtype
                )
        else:
            # Load without quantization (requires more VRAM)
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype
            )
            
        # Enable memory optimization
        pipeline.enable_model_cpu_offload()
            
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