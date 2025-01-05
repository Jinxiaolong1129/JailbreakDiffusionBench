# model_types.py
from enum import Enum, auto
from typing import Optional, Union, List
from dataclasses import dataclass
from PIL import Image
import torch
from typing import Optional, Dict, Any
from jailbreakbench.diffusion_model.diffusion_output import GenerationInput, GenerationOutput


class ModelType(str, Enum):
    TEXT_TO_IMAGE = "text2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINTING = "inpainting"

class ModelArchitecture(str, Enum):
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    KANDINSKY = "kandinsky"
    IF = "if"  # DeepFloyd IF


# base_model.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
from PIL import Image
import logging
from pathlib import Path

class DiffusionModelBase(ABC):
    """Abstract base class for all diffusion models."""
    
    def __init__(
        self,
        model_type: ModelType,
        model_arch: ModelArchitecture,
        device: str = "cuda"
    ):
        self.model_type = model_type
        self.model_arch = model_arch
        self.device = device
        self.logger = logging.getLogger(f"{model_arch}_{model_type}")
        
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize pipeline."""
        pass
    
    @abstractmethod
    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess input image for the model."""
        pass
    
    @abstractmethod
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        """Generate images from input data."""
        pass
    
    def validate_inputs(self, input_data: GenerationInput) -> None:
        """Validate input data based on model type."""
        if self.model_type == ModelType.IMAGE_TO_IMAGE and input_data.input_images is None:
            raise ValueError("Input images required for image-to-image generation")
        if self.model_type == ModelType.INPAINTING and (
            input_data.input_images is None or input_data.mask_images is None
        ):
            raise ValueError("Both input images and masks required for inpainting")

# stable_diffusion.py
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
)

class StableDiffusionModel(DiffusionModelBase):
    """Implementation for Stable Diffusion models."""
    
    def __init__(
        self,
        model_type: ModelType,
        model_arch: ModelArchitecture,
        model_path: str,
        device: str = "cuda",
    ):
        super().__init__(model_type, model_arch, device)
        self.model_path = model_path
        self.load_model()
        
    def load_model(self) -> None:
        pipeline_map = {
            ModelType.TEXT_TO_IMAGE: {
                ModelArchitecture.STABLE_DIFFUSION: StableDiffusionPipeline,
                ModelArchitecture.STABLE_DIFFUSION_XL: StableDiffusionXLPipeline,
            },
            ModelType.IMAGE_TO_IMAGE: {
                ModelArchitecture.STABLE_DIFFUSION: StableDiffusionImg2ImgPipeline,
                ModelArchitecture.STABLE_DIFFUSION_XL: StableDiffusionImg2ImgPipeline,
            },
            ModelType.INPAINTING: {
                ModelArchitecture.STABLE_DIFFUSION: StableDiffusionInpaintPipeline,
                ModelArchitecture.STABLE_DIFFUSION_XL: StableDiffusionInpaintPipeline,
            },
        }
        
        pipeline_class = pipeline_map[self.model_type][self.model_arch]
        self.pipeline = pipeline_class.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(self.device)
        
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        
    def encode_prompt(self, prompt: str, negative_prompt: Optional[str] = None) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        
        prompt_embeds = self.text_encoder(tokens)[0]
        
        if negative_prompt:
            neg_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            
            neg_embeds = self.text_encoder(neg_tokens)[0]
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds])
            
        return prompt_embeds
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if self.model_arch == ModelArchitecture.STABLE_DIFFUSION:
            target_size = 512
        else:  # SDXL
            target_size = 1024
            
        # Resize and convert to RGB
        image = image.convert("RGB")
        image = image.resize((target_size, target_size))
        
        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        self.validate_inputs(input_data)
        
        # Process all prompts
        all_images = []
        all_tokens = []
        all_latents = []
        
        for idx, prompt in enumerate(input_data.prompts):
            negative_prompt = input_data.negative_prompts[idx] if input_data.negative_prompts else None
            prompt_embeds = self.encode_prompt(prompt, negative_prompt)
            all_tokens.append(len(self.tokenizer.encode(prompt)))
            
            generation_kwargs = {
                "prompt_embeddings": prompt_embeds,
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
            }
            
            # Add image-specific arguments
            if self.model_type != ModelType.TEXT_TO_IMAGE:
                input_image = input_data.input_images[idx]
                processed_image = self.preprocess_image(input_image)
                generation_kwargs["image"] = processed_image
                
                if self.model_type == ModelType.IMAGE_TO_IMAGE:
                    generation_kwargs["strength"] = input_data.image_strength
                elif self.model_type == ModelType.INPAINTING:
                    mask_image = input_data.mask_images[idx]
                    processed_mask = self.preprocess_image(mask_image)
                    generation_kwargs["mask_image"] = processed_mask
            
            # Generate
            with torch.autocast(self.device):
                output = self.pipeline(**generation_kwargs)
                all_images.extend(output.images)
                if hasattr(output, "latents"):
                    all_latents.append(output.latents)
        
        return GenerationOutput(
            images=all_images,
            prompt_tokens=all_tokens,
            latents=all_latents if all_latents else None,
            metadata={
                "model_type": self.model_type,
                "model_arch": self.model_arch,
                "generation_params": generation_kwargs
            }
        )

# factory.py
class DiffusionModelFactory:
    """Factory class for creating diffusion models."""
    
    @staticmethod
    def create_model(
        model_type: ModelType,
        model_arch: ModelArchitecture,
        model_path: str,
        device: str = "cuda"
    ) -> DiffusionModelBase:
        if model_arch in [ModelArchitecture.STABLE_DIFFUSION, ModelArchitecture.STABLE_DIFFUSION_XL]:
            return StableDiffusionModel(model_type, model_arch, model_path, device)
        else:
            raise NotImplementedError(f"Model architecture {model_arch} not implemented yet")

# Example usage
def main():
    # Create a text-to-image model
    text2img_model = DiffusionModelFactory.create_model(
        model_type=ModelType.TEXT_TO_IMAGE,
        model_arch=ModelArchitecture.STABLE_DIFFUSION,
        model_path="runwayml/stable-diffusion-v1-5"
    )
    
    # Generate from text
    text_input = GenerationInput(
        prompts=["a beautiful sunset over mountains"],
        negative_prompts=["blurry, low quality"]
    )
    text_output = text2img_model.generate(text_input)
    
    # Create an image-to-image model
    img2img_model = DiffusionModelFactory.create_model(
        model_type=ModelType.IMAGE_TO_IMAGE,
        model_arch=ModelArchitecture.STABLE_DIFFUSION,
        model_path="runwayml/stable-diffusion-v1-5"
    )
    
    # Generate from text + image
    img2img_input = GenerationInput(
        prompts=["make it more artistic"],
        input_images=[text_output.images[0]],  # Use previous output as input
        image_strength=0.8
    )
    img2img_output = img2img_model.generate(img2img_input)

if __name__ == "__main__":
    main()