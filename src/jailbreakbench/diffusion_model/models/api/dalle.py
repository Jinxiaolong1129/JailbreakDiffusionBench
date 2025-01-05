# diffusion_model/api/dalle.py

import requests
from PIL import Image
import io
from .base import BaseAPIModel
from core.outputs import GenerationOutput

class DallEModel(BaseAPIModel):
    """DALL-E API implementation"""
    
    def __init__(self, api_key: str, model_name: str = "dall-e-3"):
        super().__init__(api_key, model_name)
        self.api_base = "https://api.openai.com/v1/images/generations"
        
    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "style": kwargs.get("style", "vivid"),
        }
        
        response_data = self._make_request(payload)
        images = self._process_response(response_data)
        
        return GenerationOutput(
            images=images,
            metadata={
                "model": self.model_name,
                "parameters": payload,
                "raw_response": response_data
            }
        )
        
    def _make_request(self, payload):
        response = requests.post(
            self.api_base, 
            headers=self.headers, 
            json=payload
        )
        response.raise_for_status()
        return response.json()
        
    def _process_response(self, response_data):
        images = []
        for data in response_data["data"]:
            img_response = requests.get(data["url"])
            img_response.raise_for_status()
            images.append(Image.open(io.BytesIO(img_response.content)))
        return images