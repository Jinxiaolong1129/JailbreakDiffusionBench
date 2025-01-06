# diffusion_model/api/stability.py
import base64
import requests
from PIL import Image
import io
from .base import BaseAPIModel
from core.outputs import GenerationOutput

class StabilityModel:
    """Stability AI API implementation for v2beta"""

    MODEL_ENDPOINTS = {
        "stability-ai-core": "core",
        "stability-ai-ultra": "ultra"
    }
    API_BASE = "https://api.stability.ai/v2beta/stable-image/generate"

    def __init__(self, api_key: str, model_name: str = "stability-ai-core"):
        """
        Initialize the Stability AI client.
        
        Args:
            api_key: Stability AI API key
            model_name: Model to use ('stability-ai-core' or 'stability-ai-ultra')
        """
        if model_name not in self.MODEL_ENDPOINTS:
            raise ValueError(f"Invalid model name. Must be one of: {list(self.MODEL_ENDPOINTS.keys())}")
            
        self.api_key = api_key
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        output_format: str = "png",
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        style_preset: Optional[str] = None,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            output_format: Output format ('webp' or 'png')
            seed: Random seed for generation
            steps: Number of diffusion steps
            cfg_scale: Classifier free guidance scale
            style_preset: Style preset to use
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            GenerationOutput containing generated images and metadata
        """
        endpoint = f"{self.API_BASE}/{self.MODEL_ENDPOINTS[self.model_name]}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*"
        }

        payload = {
            "prompt": prompt,
            "output_format": output_format
        }

        # Add optional parameters if provided
        if seed is not None:
            payload["seed"] = seed
        if steps is not None:
            payload["steps"] = steps
        if cfg_scale is not None:
            payload["cfg_scale"] = cfg_scale
        if style_preset is not None:
            payload["style_preset"] = style_preset

        # Add any additional kwargs to payload
        payload.update(kwargs)

        response = requests.post(
            endpoint,
            headers=headers,
            files={"none": ""},  # Required by the API
            data=payload
        )

        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")

        # Process the response
        image = Image.open(io.BytesIO(response.content))
        
        return GenerationOutput(
            images=[image],
            metadata={
                "model": self.model_name,
                "parameters": payload,
                "response_headers": dict(response.headers)
            }
        )

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerationOutput]:
        """
        Generate multiple images from a list of prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional parameters passed to generate()
            
        Returns:
            List of GenerationOutput objects
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    