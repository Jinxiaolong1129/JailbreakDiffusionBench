# diffusion_model/core/factory.py

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
from .wrapper import DiffusionWrapper
from .model_types import ModelType, ModelArchitecture, ModelProvider
from .outputs import GenerationInput, GenerationOutput
from ..models.T2I_model.api.dalle import DallEModel 
from ..models.T2I_model.api.stability import StabilityModel

class DiffusionFactory(DiffusionWrapper):
    """
    Factory that creates and manages both local and API-based diffusion models.
    
    Attributes:
        MODEL_REGISTRY: Maps model names to their provider info and default settings
        DEFAULT_PARAMS: Default generation parameters for each provider
    """

    MODEL_REGISTRY = {
        # Local Stable Diffusion Models
        "stable-diffusion-v1-5": {
            "provider": ModelProvider.LOCAL,
            "model_id": "sd-legacy/stable-diffusion-v1-5",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        "stable-diffusion-2": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-2",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
            }
        },   
        "stable-diffusion-3-medium": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            }
        },
        "stable-diffusion-xl-base-0.9": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-xl-base-0.9",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 40,
            "guidance_scale": 7.5,
            "refiner_steps": 20
            }
        },
        "stable-diffusion-3.5-medium": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-3.5-medium",
            "arch": ModelArchitecture.STABLE_DIFFUSION,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
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
            "safety_detector": True
            }
        },
        "flux-1-dev": {
            "provider": ModelProvider.LOCAL,
            "model_id": "black-forest-labs/FLUX.1-dev",
            "arch": ModelArchitecture.FLUX,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
            }
        },
        "cogview3": {
            "provider": ModelProvider.LOCAL,
            "model_id": "THUDM/CogView3-Plus-3B",
            "arch": ModelArchitecture.COGVIEW,
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
            "arch": ModelArchitecture.PROTEUS,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 512,
            "height": 512,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
            }
        },
        
        # New Model: PixArt Alpha
        "pixart-alpha": {
            "provider": ModelProvider.LOCAL,
            "model_id": "PixArt-alpha/PixArt-XL-2-512x512",
            "arch": ModelArchitecture.PIXART_ALPHA,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 512,
            "height": 512,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
            }
        },
        
        # New Model: PixArt Sigma
        "pixart-sigma": {
            "provider": ModelProvider.LOCAL,
            "model_id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "arch": ModelArchitecture.PIXART_SIGMA,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5
            }
        },
        
        # New Model: Stable Diffusion 3.5 Large
        "stable-diffusion-3.5-large": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-3.5-large",
            "arch": ModelArchitecture.STABLE_DIFFUSION_3,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 28,
            "guidance_scale": 4.5,
            "max_sequence_length": 512
            }
        },
        
        # New Model: Stable Diffusion 3.5 Large Turbo
        "stable-diffusion-3.5-large-turbo": {
            "provider": ModelProvider.LOCAL,
            "model_id": "stabilityai/stable-diffusion-3.5-large-turbo",
            "arch": ModelArchitecture.STABLE_DIFFUSION_3,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "max_sequence_length": 512
            }
        },
        
        # New Model: HunyuanDiT v1.2
        "hunyuan-dit-v1.2": {
            "provider": ModelProvider.TENCENT,
            "model_id": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
            "arch": ModelArchitecture.HUNYUAN_DIT,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5
            }
        },
        
        # New Model: HunyuanDiT v1.2 Distilled
        "hunyuan-dit-v1.2-distilled": {
            "provider": ModelProvider.TENCENT,
            "model_id": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
            "arch": ModelArchitecture.HUNYUAN_DIT,
            "type": ModelType.TEXT_TO_IMAGE,
            "default_params": {
            "num_inference_steps": 30,
            "guidance_scale": 6.0
            }
        },
        
        # NOTE close source models
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
        "stable-diffusion-xl-1024-v1-0": {
            "provider": ModelProvider.STABILITY,
            "model_id": "stable-diffusion-xl-1024-v1-0",
            "arch": ModelArchitecture.STABLE_DIFFUSION_XL,
            "type": ModelType.TEXT_TO_IMAGE,
        },
        "stable-diffusion-xl-turbo": {
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
        },
        ModelProvider.TENCENT: {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
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
        elif "default_params" in self.model_config:
            return self.model_config["default_params"].copy()
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
        # Google Cloud Vertex AI  Imagen 2 
        # Dalle 2
        # Dalle 3
        raise ValueError(f"Unsupported API provider: {self.provider}")

    def _create_local_model(self, **kwargs):
        """Create a local model instance."""
        if self.model_arch == ModelArchitecture.STABLE_DIFFUSION:
            from ..models.T2I_model.stable_diffusion import StableDiffusionModel
            model_path = self.MODEL_REGISTRY[self.model_name]['model_id'] # BUG
            return StableDiffusionModel(
                model_name=model_path,
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.STABLE_DIFFUSION_3:
            from ..models.T2I_model.stable_diffusion_3 import StableDiffusion3Model
            return StableDiffusion3Model(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.FLUX:
            from ..models.T2I_model.flux import FluxModel
            return FluxModel(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.COGVIEW:
            from ..models.T2I_model.cogview import CogViewModel
            return CogViewModel(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.PROTEUS:
            from ..models.T2I_model.proteus_rundiffusion import ProteusModel
            return ProteusModel(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.PIXART_ALPHA:
            from ..models.T2I_model.pixart_alpha import PixArtAlphaModel
            return PixArtAlphaModel(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.PIXART_SIGMA:
            from ..models.T2I_model.pixart_sigma import PixArtSigmaModel
            return PixArtSigmaModel(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
        elif self.model_arch == ModelArchitecture.HUNYUAN_DIT:
            from ..models.T2I_model.hunyuan_dit import HunyuanDiTModel
            return HunyuanDiTModel(
                model_name=self.model_config["model_id"],
                device=self.device,
                **kwargs
            )
            
        raise ValueError(f"Unsupported local model architecture: {self.model_arch}")

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> GenerationOutput:
        """
        Generate outputs from the given prompts.
        
        Args:
            prompts: A single text prompt or a list of text prompts
            **kwargs: Override default generation parameters
            
        Returns:
            GenerationOutput containing the generated images and metadata
        """
        try:
            if isinstance(prompts, str):
                prompts = [prompts]
            elif not isinstance(prompts, list):
                raise TypeError("prompts must be a string or a list of strings")
            
            params = self.generation_params.copy()
            params.update(kwargs)
            
            if self.is_api_model:
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
            raise ValueError(f"Generation failed: {str(e)}")

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