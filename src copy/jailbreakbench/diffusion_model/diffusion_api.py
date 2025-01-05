from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List
from enum import Enum
import time
import requests
from PIL import Image
import io
import base64


class ModelProvider(Enum):
    DALLE = "dalle"
    STABILITY = "stability"
    LEONARDO = "leonardo"

DIFFUSION_MODEL_NAMES = {
    "dalle-3": (ModelProvider.DALLE, "dall-e-3"),
    "dalle-2": (ModelProvider.DALLE, "dall-e-2"),
    "sdxl": (ModelProvider.STABILITY, "stable-diffusion-xl-1024-v1-0"),
    "sdxl-turbo": (ModelProvider.STABILITY, "stable-diffusion-xl-turbo"),
    "leonardo-sdxl": (ModelProvider.LEONARDO, "sdxl"),
    "leonardo-creative": (ModelProvider.LEONARDO, "creative")
}

# Default generation parameters for each provider
DEFAULT_GENERATION_PARAMS = {
    ModelProvider.DALLE: {
        "size": "1024x1024",
        "quality": "standard",
        "style": "vivid",
    },
    ModelProvider.STABILITY: {
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "cfg_scale": 7.0,
    },
    ModelProvider.LEONARDO: {
        "width": 1024,
        "height": 1024,
        "negative_prompt": "",
    }
}

@dataclass
class DiffusionOutput:
    """Container for diffusion model outputs"""
    images: List[Image.Image]
    prompt_tokens: Optional[int] = None
    generation_params: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None

class DiffusionLiteDM:
    """Unified interface for different diffusion models"""
    _N_RETRIES: int = 5
    _RETRY_DELAY: float = 1.0

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the DiffusionLiteDM with specified model name and API key.
        
        Args:
            model_name: Name of the model (e.g., "dalle-3", "sdxl", "leonardo-sdxl")
            api_key: API key for the corresponding service
        """
        if model_name not in DIFFUSION_MODEL_NAMES:
            raise ValueError(f"Unknown model name: {model_name}. Available models: {list(DIFFUSION_MODEL_NAMES.keys())}")
            
        self.model_name = model_name
        self.provider, self.provider_model = DIFFUSION_MODEL_NAMES[model_name]
        self.api_key = api_key
        
        # Initialize provider-specific client
        if self.provider == ModelProvider.DALLE:
            self.client = DallEClient(api_key, self.provider_model)
        elif self.provider == ModelProvider.STABILITY:
            self.client = StabilityClient(api_key, self.provider_model)
        elif self.provider == ModelProvider.LEONARDO:
            self.client = LeonardoClient(api_key, self.provider_model)
            
        # Set default generation parameters
        self.generation_params = DEFAULT_GENERATION_PARAMS[self.provider].copy()

    def generate(self, prompt: str, **kwargs) -> DiffusionOutput:
        """
        Generate images using the specified model.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional provider-specific parameters
        
        Returns:
            DiffusionOutput containing generated images and metadata
        """
        # Update generation parameters with any provided kwargs
        generation_params = self.generation_params.copy()
        generation_params.update(kwargs)
        
        return self.client.generate(prompt, **generation_params)

class BaseClient:
    """Base class for provider-specific clients"""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _handle_retry(self, func, *args, **kwargs):
        """Implements retry logic with exponential backoff"""
        for attempt in range(DiffusionLiteDM._N_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == DiffusionLiteDM._N_RETRIES - 1:
                    raise e
                delay = DiffusionLiteDM._RETRY_DELAY * (2 ** attempt)
                time.sleep(delay)
        return None

class DallEClient(BaseClient):
    """DALL-E API client"""
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.api_base = "https://api.openai.com/v1/images/generations"

    def generate(self, prompt: str, **kwargs) -> DiffusionOutput:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "style": kwargs.get("style", "vivid"),
        }

        def _make_request():
            response = requests.post(self.api_base, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

        response_data = self._handle_retry(_make_request)
        
        images = []
        for data in response_data["data"]:
            img_response = requests.get(data["url"])
            img_response.raise_for_status()
            images.append(Image.open(io.BytesIO(img_response.content)))

        return DiffusionOutput(
            images=images,
            generation_params=payload,
            raw_response=response_data
        )

class StabilityClient(BaseClient):
    """Stability AI API client"""
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.api_base = "https://api.stability.ai/v1/generation"

    def generate(self, prompt: str, **kwargs) -> DiffusionOutput:
        endpoint = f"{self.api_base}/{self.model}/text-to-image"
        
        payload = {
            "text_prompts": [{"text": prompt}],
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 1024),
            "samples": kwargs.get("samples", 1),
            "steps": kwargs.get("steps", 50),
            "cfg_scale": kwargs.get("cfg_scale", 7.0),
        }

        def _make_request():
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()

        response_data = self._handle_retry(_make_request)
        
        images = []
        for artifact in response_data["artifacts"]:
            img_data = base64.b64decode(artifact["base64"])
            images.append(Image.open(io.BytesIO(img_data)))

        return DiffusionOutput(
            images=images,
            generation_params=payload,
            raw_response=response_data
        )

class LeonardoClient(BaseClient):
    """Leonardo AI API client"""
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.api_base = "https://cloud.leonardo.ai/api/rest/v1/generations"

    def generate(self, prompt: str, **kwargs) -> DiffusionOutput:
        payload = {
            "prompt": prompt,
            "modelId": self.model,
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 1024),
            "num_images": kwargs.get("num_images", 1),
            "negative_prompt": kwargs.get("negative_prompt", ""),
        }

        def _make_request():
            response = requests.post(self.api_base, headers=self.headers, json=payload)
            response.raise_for_status()
            generation_data = response.json()
            
            generation_id = generation_data["sdGenerationJob"]["generationId"]
            while True:
                status_response = requests.get(
                    f"{self.api_base}/{generation_id}",
                    headers=self.headers
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                
                if status_data["generations_by_pk"]["status"] == "COMPLETE":
                    return status_data
                elif status_data["generations_by_pk"]["status"] == "FAILED":
                    raise Exception("Generation failed")
                    
                time.sleep(1)

        response_data = self._handle_retry(_make_request)
        
        images = []
        for generated in response_data["generations_by_pk"]["generated_images"]:
            img_response = requests.get(generated["url"])
            img_response.raise_for_status()
            images.append(Image.open(io.BytesIO(img_response.content)))

        return DiffusionOutput(
            images=images,
            generation_params=payload,
            raw_response=response_data
        )

if __name__ == "__main__":
    # Using DALL-E 3
    dalle_dm = DiffusionLiteDM(
        model_name="dalle-3",
        api_key="your-openai-api-key"
    )
    dalle_output = dalle_dm.generate(
        prompt="A serene mountain landscape at sunset",
        size="1024x1024"
    )

    # Using Stability AI
    stability_dm = DiffusionLiteDM(
        model_name="sdxl",
        api_key="your-stability-api-key"
    )
    stability_output = stability_dm.generate(
        prompt="A futuristic cityscape with flying cars",
        samples=1
    )

    # Using Leonardo AI
    leonardo_dm = DiffusionLiteDM(
        model_name="leonardo-sdxl",
        api_key="your-leonardo-api-key"
    )
    leonardo_output = leonardo_dm.generate(
        prompt="An abstract painting with vibrant colors",
        num_images=1
    )