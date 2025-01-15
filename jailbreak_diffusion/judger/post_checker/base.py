# base.py
from abc import ABC, abstractmethod
from typing import Union, List
from pathlib import Path
from PIL import Image

class BaseImageChecker(ABC):
    """Base class for all image content checkers."""
    
    def __init__(self):
        """Initialize the checker."""
        pass

    @abstractmethod
    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """Process image(s) and return check result(s).
        
        Args:
            image_input: Input image(s), can be:
                - str or Path: path to image file
                - PIL.Image: PIL image object
                - List of the above: multiple images
                
        Returns:
            bool or List[bool]: True if inappropriate content detected.
            Returns a single bool for single input, list of bools for list input.
            
        Raises:
            ValueError: If input format is not supported or file doesn't exist
        """
        pass