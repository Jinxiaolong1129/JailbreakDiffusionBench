# diffusion_model/core/factory.py

from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
from .wrapper import DiffusionWrapper
from .model_types import ModelType, ModelArchitecture, ModelProvider
from .outputs import GenerationInput, GenerationOutput
from models.api import DallEModel, StabilityModel, LeonardoModel

class DiffusionFactory(DiffusionWrapper):
    """
    Factory that creates and manages both local and API-based diffusion models.
    
    Attributes:
        MODEL_REGISTRY: Maps model names to their provider info and default settings
        DEFAULT_PARAMS: Default generation parameters for each provider
    """

    MODEL_REGISTRY = {
        # Local Stable Diffusion Models
        "stable-diffusion-3": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-3-medium",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            }
        },
        "stable-diffusion-xl-refiner": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "arch": ModelArchitecture.STABLE_DIFFUSION_XL,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "refiner_steps": 20
            }
        },
        "stable-diffusion-safe": {
            "provider": ModelProvider.LOCAL,
            "model_id": "AIML-TUDA/stable-diffusion-safe",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "safety_checker": True
            }
        },
        "cogview3": {
            "provider": ModelProvider.LOCAL,
            "model_id": "THUDM/CogView3-Plus-3B",
            "arch": ModelArchitecture.IF,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50
            }
        },
        "min-dalle": {
            "provider": ModelProvider.LOCAL,
            "model_id": "kuprel/min-dalle",
            "arch": ModelArchitecture.DALLE,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
                "width": 256,
                "height": 256,
                "supercondition_factor": 16
            }
        },
        "proteus-rundiffusion": {
            "provider": ModelProvider.LOCAL,
            "model_id": "dataautogpt3/Proteus-RunDiffusion",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            }
        },
        # DALL-E Models
        "dalle-3": {
            "provider": ModelProvider.OPENAI,
            "model_id": "dall-e-3",
            "arch": ModelArchitecture.DALLE,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        "dalle-2": {
            "provider": ModelProvider.OPENAI,
            "model_id": "dall-e-2",
            "arch": ModelArchitecture.DALLE,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        
        # Stability AI Models
        "sdxl": {
            "provider": ModelProvider.STABILITY,
            "model_id": "stable-diffusion-xl-1024-v1-0",
            "arch": ModelArchitecture.STABLE_DIFFUSION_XL,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        "sdxl-turbo": {
            "provider": ModelProvider.STABILITY,
            "model_id": "stable-diffusion-xl-turbo",
            "arch": ModelArchitecture.STABLE_DIFFUSION_XL,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        
        # Leonardo AI Models
        "leonardo-sdxl": {
            "provider": ModelProvider.LEONARDO,
            "model_id": "sdxl",
            "arch": ModelArchitecture.STABLE_DIFFUSION_XL,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        "leonardo-creative": {
            "provider": ModelProvider.LEONARDO,
            "model_id": "creative",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
        },
    }

    DEFAULT_PARAMS = {
        ModelProvider.OPENAI: {
            "size": "1024x1024",
            "quality": "standard",
            "style": "vivid",
            "n": 1,
        },
        ModelProvider.STABILITY: {
            "width": 1024,
            "height": 1024,
            "samples": 1,
            "steps": 50,
            "cfg_scale": 7.0,
        },
        ModelProvider.LEONARDO: {
            "width": 1024,
            "height": 1024,
            "negative_prompt": "",
            "num_images": 1,
        }
    }

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        save_dir: str = "outputs",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the DiffusionFactory.
        
        Args:
            model_name: Name of the model to use (must be in MODEL_REGISTRY)
            api_key: API key for cloud services (required for API models)
            save_dir: Directory to save generation outputs
            device: Device to run local models on ('cuda' or 'cpu')
            **kwargs: Additional model-specific initialization parameters
        """
        super().__init__(save_dir=save_dir, device=device)
        self.logger = logging.getLogger(__name__)
        
        # Validate and set up model configuration
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_REGISTRY.keys())}")
            
        self.model_config = self.MODEL_REGISTRY[model_name]
        self.model_name = model_name
        self.api_key = api_key
        
        # Determine if this is an API model
        self.is_api_model = self.model_config["provider"] != ModelProvider.LOCAL
        if self.is_api_model and not api_key:
            raise ValueError(f"API key required for {model_name}")
            
        # Set up model parameters
        self.provider = self.model_config["provider"]
        self.model_type = self.model_config["type"]
        self.model_arch = self.model_config["arch"]
        self.generation_params = self._get_default_params()
        
        # Initialize the model
        self.model = self._create_model(**kwargs)
        self.logger.info(f"Initialized {model_name} ({self.provider})")

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default generation parameters for the current model."""
        if self.is_api_model:
            return self.DEFAULT_PARAMS[self.provider].copy()
        return {}

    def _create_model(self, **kwargs):
        """Create and return the appropriate model instance."""
        if self.is_api_model:
            return self._create_api_model(**kwargs)
        return self._create_local_model(**kwargs)

    def _create_api_model(self, **kwargs):
        """Create an API-based model instance."""
        model_id = self.model_config["model_id"]
        
        if self.provider == ModelProvider.OPENAI:
            return DallEModel(self.api_key, model_id)
        elif self.provider == ModelProvider.STABILITY:
            return StabilityModel(self.api_key, model_id)
        elif self.provider == ModelProvider.LEONARDO:
            return LeonardoModel(self.api_key, model_id)
        
        raise ValueError(f"Unsupported API provider: {self.provider}")

    def _create_local_model(self, **kwargs):
        """Create a local model instance."""
        if self.model_arch == ModelArchitecture.STABLE_DIFFUSION:
            from models.model.stable_diffusion import StableDiffusionModel
            return StableDiffusionModel(
                model_path=self.model_name,
                device=self.device,
                is_xl="xl" in self.model_name.lower(),
                **kwargs
            )
        # Add support for other local model architectures here
        raise ValueError(f"Unsupported local model architecture: {self.model_arch}")

    def generate(self, prompts: List[str], **kwargs) -> GenerationOutput:
        """
        Generate outputs from the given prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Override default generation parameters
            
        Returns:
            GenerationOutput containing the generated images and metadata
        """
        try:
            # Merge default and custom parameters
            params = self.generation_params.copy()
            params.update(kwargs)
            
            if self.is_api_model:
                # Handle API model generation
                outputs = []
                for prompt in prompts:
                    output = self.model.generate(prompt, **params)
                    outputs.append(output)
                return self._merge_outputs(outputs)
            else:
                # Handle local model generation
                input_data = GenerationInput(
                    prompts=prompts,
                    **params
                )
                return self.model.generate(input_data)
                
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise

    def _merge_outputs(self, outputs: List[GenerationOutput]) -> GenerationOutput:
        """Merge multiple generation outputs into a single output."""
        all_images = []
        all_metadata = {}
        
        for output in outputs:
            if output.images:
                all_images.extend(output.images)
            if output.metadata:
                all_metadata.update(output.metadata)
                
        return GenerationOutput(
            images=all_images,
            metadata=all_metadata
        )

    def validate_inputs(self, prompts: List[str], **kwargs):
        """Validate generation inputs before processing."""
        if not prompts:
            raise ValueError("At least one prompt is required")
            
        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list of strings")
            
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise TypeError("Each prompt must be a string")
                
        # Add model-specific validation here
        if self.model_type == ModelType.IMAGE_TO_IMAGE:
            if "input_image" not in kwargs:
                raise ValueError("input_image required for image-to-image generation")
            
