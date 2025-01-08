# diffusion_model/models/api/base.py

from abc import ABC, abstractmethod
import time
import requests
import logging
from typing import Dict, Any, Optional
from ...core.outputs import GenerationOutput

class BaseAPIModel(ABC):
    """Base class for API-based models"""
    
    MAX_RETRIES: int = 5
    RETRY_DELAY: float = 1.0
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """Generate outputs from prompt"""
        pass
    
    def _make_request(self, url: str, payload: dict, method: str = "POST") -> dict:
        """Make API request with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == self.MAX_RETRIES - 1:
                    self.logger.error(f"API request failed after {self.MAX_RETRIES} attempts: {str(e)}")
                    raise
                
                delay = self.RETRY_DELAY * (2 ** attempt)
                self.logger.warning(f"Request failed, retrying in {delay}s... ({attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(delay)
    
    def _validate_response(self, response: dict) -> None:
        """Validate API response"""
        if not response:
            raise ValueError("Empty response from API")
        if "error" in response:
            raise ValueError(f"API error: {response['error']}")