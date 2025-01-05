# diffusion_model/api/stability.py
# 
import base64
import requests
from PIL import Image
import io
from .base import BaseAPIModel
from core.outputs import GenerationOutput

class StabilityModel(BaseAPIModel):
    """Stability AI API implementation"""
    
    def __init__(self, api_key: str, model_name: str = "stable-diffusion-xl-1024-v1-0"):
        super().__init__(api_key, model_name)
        self.api_base = "https://api.stability.ai/v1/generation"
        
    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        endpoint = f"{self.api_base}/{self.model_name}/text-to-image"
        
        payload = {
            "text_prompts": [{"text": prompt}],
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 1024),
            "samples": kwargs.get("samples", 1),
            "steps": kwargs.get("steps", 50),
            "cfg_scale": kwargs.get("cfg_scale", 7.0),
        }
        
        response_data = self._make_request(endpoint, payload)
        images = self._process_response(response_data)
        
        return GenerationOutput(
            images=images,
            metadata={
                "model": self.model_name,
                "parameters": payload,
                "raw_response": response_data
            }
        )