# jailbreak_diffusion/judger/post_checker/base.py
from abc import ABC, abstractmethod
from typing import Union, List, Optional
from pathlib import Path
from PIL import Image
import io

class BaseImageChecker(ABC):
    """Base class for all image content checkers."""
    
    def __init__(self):
        """Initialize the checker."""
        pass

    @abstractmethod
    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """Process image(s) and return check result(s)."""
        pass
        
    def check(self, image_input: Union[str, Path, Image.Image, 
                                     List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Unified check method for all detectors.
        Returns True if unsafe content detected.
        """
        return self(image_input)

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Helper method to load image from various input types."""
        if isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError("Unsupported image input type")

    def _process_image_list(self, images: List[Union[str, Path, Image.Image]]) -> List[Image.Image]:
        """Helper method to process a list of images."""
        return [self._load_image(img) for img in images]