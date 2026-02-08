# diffusion_model/core/model_types.py

from enum import Enum, auto

class ModelType(str, Enum):
    """Types of generation tasks supported by models"""
    TEXT_TO_IMAGE = "text2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINTING = "inpainting"
    TEXT_TO_VIDEO = "text2video"
    IMAGE_TO_VIDEO = "img2video"

class ModelArchitecture(str, Enum):
    """Supported model architectures"""
    STABLE_DIFFUSION = "stable_diffusion"
    STABLE_DIFFUSION_XL = "stable_diffusion_xl"
    STABLE_DIFFUSION_3 = "stable_diffusion_3"
    FLUX = "flux"
    DALLE = "dalle"
    ANIMATED_DIFF = "animated_diff"
    STABLE_VIDEO = "stable_video"
    MODELSCOPE = "modelscope"
    COGVIEW = "cogview"
    PROTEUS = "proteus"
    PIXART_ALPHA = "pixart_alpha"
    PIXART_SIGMA = "pixart_sigma"
    HUNYUAN_DIT = "hunyuan_dit"
    
    
class ModelProvider(str, Enum):
    """Model providers/sources"""
    OPENAI = "openai"
    STABILITY = "stability"
    LEONARDO = "leonardo"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For local models