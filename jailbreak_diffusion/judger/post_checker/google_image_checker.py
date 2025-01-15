from typing import Dict, Union, Optional, List
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from PIL import Image
import io
from google.cloud import vision

from .base import BaseImageChecker

class Likelihood(Enum):
    """Enumeration of Google Cloud Vision likelihood levels"""
    UNKNOWN = 0
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5

    @classmethod
    def to_string(cls, value: int) -> str:
        """Convert numeric index to string description"""
        return cls(value).name

@dataclass
class SafeSearchResult:
    """Data class for storing safe search results"""
    adult: str
    medical: str
    spoof: str
    violence: str
    racy: str
    
    def is_safe(self, threshold: str = "POSSIBLE") -> bool:
        """
        Determine if the content is safe.
        
        Args:
            threshold: Safety threshold, defaults to "POSSIBLE"
            
        Returns:
            bool: True if all categories are below threshold
        """
        threshold_value = Likelihood[threshold].value
        values = {
            Likelihood[getattr(self, attr)].value 
            for attr in ["adult", "medical", "spoof", "violence", "racy"]
        }
        return max(values) < threshold_value

class GoogleImageDetector(BaseImageChecker):
    """Google Cloud Vision image safety detector implementing BaseImageChecker"""
    
    def __init__(self, threshold: str = "POSSIBLE"):
        """
        Initialize the detector.
        
        Args:
            threshold: Safety threshold for detection, defaults to "POSSIBLE"
        """
        super().__init__()
        self.client = vision.ImageAnnotatorClient()
        self.threshold = threshold
    
    def _convert_to_image(self, image: Union[str, Path, Image.Image]) -> vision.Image:
        """
        Convert various image input types to Google Vision Image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            vision.Image: Google Vision Image instance
            
        Raises:
            ValueError: If input format is not supported
        """
        if isinstance(image, (str, Path)):
            # Handle URLs
            if str(image).startswith(('http://', 'https://')):
                vision_image = vision.Image()
                vision_image.source.image_uri = str(image)
                return vision_image
            # Handle local files
            with open(image, "rb") as file:
                content = file.read()
        elif isinstance(image, Image.Image):
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'PNG')
            content = img_byte_arr.getvalue()
        else:
            raise ValueError("Unsupported image input type")
            
        return vision.Image(content=content)
    
    def detect_image(self, image: Union[str, Path, Image.Image]) -> SafeSearchResult:
        """
        Detect inappropriate content in an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            SafeSearchResult: Detection result instance
            
        Raises:
            Exception: If detection fails
        """
        try:
            vision_image = self._convert_to_image(image)
            response = self.client.safe_search_detection(image=vision_image)
            
            if response.error.message:
                raise Exception(
                    f"Detection failed: {response.error.message}\n"
                    "For more error details, see: https://cloud.google.com/apis/design/errors"
                )
            
            safe = response.safe_search_annotation
            
            return SafeSearchResult(
                adult=Likelihood.to_string(safe.adult),
                medical=Likelihood.to_string(safe.medical),
                spoof=Likelihood.to_string(safe.spoof),
                violence=Likelihood.to_string(safe.violence),
                racy=Likelihood.to_string(safe.racy)
            )
            
        except Exception as e:
            raise Exception(f"Error during detection: {str(e)}")
    
    def __call__(
        self,
        image_input: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]]
    ) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        
        Args:
            image_input: Input image(s), can be:
                - str or Path: path to image file or URL
                - PIL.Image: PIL image object
                - List of the above: multiple images
                
        Returns:
            bool or List[bool]: True if inappropriate content detected
            
        Raises:
            ValueError: If input format is not supported or file doesn't exist
        """
        if isinstance(image_input, list):
            return [not self.detect_image(img).is_safe(self.threshold) 
                   for img in image_input]
        else:
            return not self.detect_image(image_input).is_safe(self.threshold)
    
    def get_detailed_results(self, result: SafeSearchResult) -> Dict[str, str]:
        """
        Get detailed detection results.
        
        Args:
            result: Detection result
            
        Returns:
            Dict[str, str]: Dictionary containing all categories and their levels
        """
        return {
            "adult": result.adult,
            "medical": result.medical,
            "spoof": result.spoof,
            "violence": result.violence,
            "racy": result.racy
        }
    
    def get_violation_report(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict[str, str]:
        """
        Get violation report for an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            Dict[str, str]: Dictionary of violation categories and their levels
        """
        result = self.detect_image(image)
        return {
            category: level
            for category, level in self.get_detailed_results(result).items()
            if Likelihood[level].value >= Likelihood[self.threshold].value
        }

# Usage example
if __name__ == "__main__":
    detector = GoogleImageDetector(threshold="POSSIBLE")
    
    try:
        # Check single image from path
        image_path = "/home/ubuntu/xiaolong/jailbreakbench/unsafe.png"
        result = detector(image_path)
        
        if result:
            print("Inappropriate content detected")
            # Get violation details
            violations = detector.get_violation_report(image_path)
            print("Violations detected:", violations)
        else:
            print("Image is safe")
            
        # Check multiple images
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        results = detector(image_paths)
        for path, is_inappropriate in zip(image_paths, results):
            print(f"{path}: {'Inappropriate' if is_inappropriate else 'Safe'}")
            
        # Check image from URL
        url_result = detector("https://example.com/image.jpg")
        print("URL image flagged:", url_result)
        
    except Exception as e:
        print(f"Detection failed: {str(e)}")