# test_sd_models.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import torch
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import logging
from diffusers import StableDiffusion3Pipeline
from diffusers import DiffusionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationInput:
    prompts: list[str]
    negative_prompt: str = None
    extra_params: Dict[str, Any] = None

@dataclass
class GenerationOutput:
    images: list
    metadata: Dict[str, Any]

class StableDiffusionModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
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
            use_safetensors=True,
            device_map="balanced"
        )
        return pipeline
        
    def generate(self, input_data: GenerationInput) -> GenerationOutput:
        params = input_data.extra_params or {}
        all_images = []
        start_time = time.time()
        
        generation_params = {
            "prompt": None,  
            "negative_prompt": input_data.negative_prompt,
            "num_inference_steps": params.get("num_inference_steps", 50),
            # "guidance_scale": params.get("guidance_scale", 7.5),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
        }
        
        for prompt in input_data.prompts:
            generation_params["prompt"] = prompt
            print(generation_params)
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

def test_model(model_name: str, save_dir: str = "outputs", test_prompts: list[str] = None):
    """Test a specific SD model"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    if test_prompts is None:
        test_prompts = [
            "A capybara holding a sign that reads Hello World",
        ]
    
    logger.info(f"\nTesting model: {model_name}")
    
    # try:
    # Initialize model
    model = StableDiffusionModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16
    )
    logger.info("Model loaded successfully")
    
    # Test each prompt
    for idx, prompt in enumerate(test_prompts):
        logger.info(f"\nGenerating test image {idx+1}")
        logger.info(f"Prompt: {prompt}")
        
        input_data = GenerationInput(
            prompts=[prompt],
            negative_prompt="low quality, blurry, distorted",
            extra_params={
                "num_inference_steps": 50,
                "width": 1024,
                "height": 1024
            }
        )
        
        output = model.generate(input_data)
        
        # Save images
        for i, image in enumerate(output.images):
            save_path = save_dir / f"{model_name.split('/')[-1]}_test{idx+1}.png"
            image.save(save_path)
            logger.info(f"Saved image to {save_path}")
            logger.info(f"Generation time: {output.metadata['generation_time']:.2f}s")
                
    # except Exception as e:
    #     logger.error(f"Test failed: {str(e)}")

def main():
    # Models to test
    models = [
        # "stabilityai/stable-diffusion-2",
        # "stabilityai/stable-diffusion-3-medium-diffusers",
        # "stabilityai/stable-diffusion-3.5-medium",
        # "stabilityai/stable-diffusion-xl-base-0.9",
        # "CompVis/stable-diffusion-v1-4",
        # "AIML-TUDA/stable-diffusion-safe",
        # "black-forest-labs/FLUX.1-dev"
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    for model_name in models:
        test_model(model_name)

if __name__ == "__main__":
    main()