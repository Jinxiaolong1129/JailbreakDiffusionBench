# __init__.py

from .MultiheadDetector import MultiheadDetector
from .Q16 import Q16Detector, FinetunedQ16Detector
from .SD_safety_checker import SafetyCheckerDetector
from .openai_image_checker import OpenAIImageDetector
from .azure_image_checker import AzureImageDetector
from .google_image_checker import GoogleImageDetector
from .aws_image_checker import AWSImageDetector

__all__ = [
    'MultiheadDetector',
    'Q16Detector', 
    'FinetunedQ16Detector',
    'SafetyCheckerDetector',
    'OpenAIImageDetector',
    'AzureImageDetector', 
    'GoogleImageDetector',
    'AWSImageDetector'
]

# Version info
__version__ = '1.0.0'