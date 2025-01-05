# diffusion_model/models/model/min_dalle.py

from min_dalle import MinDalle
import torch
from PIL import Image
from .base import BaseDiffusionModel
from core.outputs import GenerationInput, GenerationOutput
import time


class MinDalleModel(BaseDiffusionModel):
    """Implementation for Min-DALLE model"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        super().__init__(model_path, device, torch_dtype)
        self.supercondition_factor = kwargs.get("supercondition_factor", 16)
        
    def load_model(self):
        return MinDalle(
            models_root="./models",
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device=self.device,
            is_mega=True,
            is_reusable=True
        )
        
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        all_images = []
        start_time = time.time()
        
        for prompt in input_data.prompts:
            # Generate image
            generated_images = self.model.generate_images(
                text=prompt,
                seed=input_data.seed,
                grid_size=1,
                is_seamless=False,
                temperature=1,
                top_k=256,
                supercondition_factor=self.supercondition_factor,
                is_verbose=False
            )
            
            # Convert to PIL Image
            image = Image.fromarray(generated_images[0])
            
            # Resize if needed
            if "width" in input_data.extra_params and "height" in input_data.extra_params:
                image = image.resize((
                    input_data.extra_params["width"],
                    input_data.extra_params["height"]
                ))
                
            all_images.append(image)
            
        generation_time = time.time() - start_time
                
        return GenerationOutput(
            images=all_images,
            metadata={
                "model": self.model_path,
                "parameters": input_data.to_dict(),
                "generation_time": generation_time
            }
        )
