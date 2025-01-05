# diffusion_wrapper.py
from diffusers import DiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer
import torch
from pathlib import Path
import json
import datetime
import abc

class DiffusionWrapper(abc.ABC):
    """Base class for diffusion model wrapper"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logs_path = Path("logs")
        self.logs_path.mkdir(exist_ok=True)
        
    @abc.abstractmethod
    def generate_images(self, prompts: list[str], **kwargs) -> list[torch.Tensor]:
        """Generate images from text prompts"""
        pass
        
    def log_generations(self, prompts: list[str], images: list[torch.Tensor], 
                       phase: str = "train"):
        """Log generation results"""
        timestamp = datetime.datetime.now().isoformat()
        log_path = self.logs_path / phase / f"{self.model_name}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save images and metadata
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            image_path = log_path.parent / f"image_{timestamp}_{i}.png"
            # Save image using PIL
            image_pil = self.tensor_to_pil(image)
            image_pil.save(image_path)
            
            log_entry = {
                "timestamp": timestamp,
                "prompt": prompt,
                "image_path": str(image_path)
            }
            
            with open(log_path, "a+") as f:
                json.dump(log_entry, f)
                f.write("\n")
                
    @staticmethod
    def tensor_to_pil(tensor):
        """Convert tensor to PIL image"""
        return tensor.cpu().permute(1,2,0).numpy()




class StableDiffusionWrapper(DiffusionWrapper):
    """Wrapper for Stable Diffusion models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            scheduler=DDIMScheduler.from_pretrained(model_name, subfolder="scheduler"),
            torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        
    def generate_images(self, prompts: list[str], 
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5,
                       **kwargs) -> list[torch.Tensor]:
        """Generate images using Stable Diffusion"""
        images = []
        for prompt in prompts:
            with torch.autocast(self.device):
                image = self.pipeline(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    **kwargs
                ).images[0]
            images.append(image)
        return images
        
    def update_scheduler(self, scheduler_type: str):
        """Update the scheduler type"""
        self.pipeline.scheduler = scheduler_type.from_config(
            self.pipeline.scheduler.config
        )