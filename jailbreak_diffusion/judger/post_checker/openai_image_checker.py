# jailbreak_diffusion/judger/post_checker/openai_image_checker.py
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io
import os
from openai import OpenAI

from .base import BaseImageChecker
# NOTE finish testing
@dataclass
class ModerationResult:
    """Data class for storing moderation results"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Dict[str, List[str]]

    def get_violation_details(self) -> Dict[str, float]:
        """Get details of violated categories."""
        scores_dict = self.category_scores.dict()
        return {
            category: scores_dict[category]
            for category in scores_dict
        }

class OpenAIImageDetector(BaseImageChecker):
    """OpenAI image content moderation detector implementing BaseImageChecker"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "omni-moderation-latest"):
        """
        Initialize the detector.
        
        Args:
            api_key: OpenAI API key. If None, fetched from OPENAI_API_KEY environment variable
        """
        super().__init__()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Must provide api_key or set OPENAI_API_KEY environment variable"
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = "omni-moderation-latest"
        
    def _convert_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Convert various image input types to base64 string.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            str: Base64 encoded image data with data URL prefix
            
        Raises:
            ValueError: If input format is not supported
        """
        try:
            if isinstance(image, (str, Path)):
                # If URL, return as is
                if str(image).startswith(('http://', 'https://')):
                    return str(image)
                # Otherwise read file
                with open(image, "rb") as file:
                    image_bytes = file.read()
            elif isinstance(image, Image.Image):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'PNG')
                image_bytes = img_byte_arr.getvalue()
            else:
                raise ValueError("Unsupported image input type")
                
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/png;base64,{base64_image}"
        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {str(e)}")
    
    def detect_image(
        self,
        image: Union[str, Path, Image.Image],
        text: Optional[str] = None
    ) -> ModerationResult:
        """
        Detect inappropriate content in an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            text: Optional accompanying text content
            
        Returns:
            ModerationResult: Detection result instance
            
        Raises:
            Exception: If detection fails
        """
        try:
            # Convert image to base64 or get URL
            image_url = self._convert_to_base64(image)
            
            # Build input
            inputs = []
            if text:
                inputs.append({"type": "text", "text": text})
            
            inputs.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            
            # Call API
            response = self.client.moderations.create(
                model=self.model,
                input=inputs
            )
            
            # Get first result
            result = response.results[0]
            
            return ModerationResult(
                flagged=result.flagged,
                categories=result.categories,
                category_scores=result.category_scores,
                category_applied_input_types=result.category_applied_input_types
            )
            
        except Exception as e:
            raise Exception(f"OpenAI detection failed: {str(e)}")
    
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
            return [self.detect_image(img).flagged for img in image_input]
        else:
            return self.detect_image(image_input).flagged
    
    def get_violation_report(
        self,
        image: Union[str, Path, Image.Image],
        text: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get detailed violation report for an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            text: Optional accompanying text content
            
        Returns:
            Dict[str, float]: Violation categories and their severity scores
        """
        result = self.detect_image(image, text)
        return result.get_violation_details()

if __name__ == "__main__":
    detector = OpenAIImageDetector()
    
    try:
        # Check single image
        image_path = "unsafe.png"
        result = detector.check(image_path)
        print(f"Image is {'unsafe' if result else 'safe'}")
        
        if result:
            violations = detector.get_violation_report(image_path)
            print("Violations detected:", violations)
            
        # Check multiple images
        image_paths = ["unsafe.png", "unsafe.png", "unsafe.png"]
        results = detector.check(image_paths)
        for path, is_unsafe in zip(image_paths, results):
            print(f"{path}: {'Unsafe' if is_unsafe else 'Safe'}")
            
        # Check image with text
        text_result = detector.get_violation_report(
            "unsafe.png",
            text="Check this image content"
        )
        print("Text context violations:", text_result)
        
    except Exception as e:
        print(f"Detection failed: {str(e)}")