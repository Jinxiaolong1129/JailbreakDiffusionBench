from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io
import os
from openai import OpenAI

from .base import BaseImageChecker

# TODO finish testing

@dataclass
class ModerationResult:
    """Data class for storing moderation results"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Dict[str, List[str]]

class OpenAIImageDetector(BaseImageChecker):
    """OpenAI image content moderation detector implementing BaseImageChecker"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the detector.
        
        Args:
            api_key: OpenAI API key. If None, fetched from OPENAI_API_KEY environment variable
        """
        super().__init__()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', api_key))
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
        if isinstance(image, (str, Path)):
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
            # Convert image to base64 if it's not already a URL
            if isinstance(image, str) and (image.startswith('http://') or image.startswith('https://')):
                image_url = image
            else:
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
            raise Exception(f"Detection failed: {str(e)}")
    
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
    
    def get_violation_details(self, result: ModerationResult) -> Dict[str, float]:
        """
        Get detailed violation information.
        
        Args:
            result: Detection result
            
        Returns:
            Dict[str, float]: Dictionary of violation categories and their confidence scores
        """
        return {
            category: score
            for category, score in result.category_scores.items()
            if result.categories[category]
        }


# Usage example
if __name__ == "__main__":
    detector = OpenAIImageDetector()
    
    try:
        # Check single image from path
        image_path = "/home/ubuntu/xiaolong/jailbreakbench/unsafe.png"
        result = detector(image_path)
        print(result)
            
        image_paths = ["/home/ubuntu/xiaolong/jailbreakbench/unsafe.png", "/home/ubuntu/xiaolong/jailbreakbench/unsafe.png"]
        results = detector(image_paths)
        
        for path, is_inappropriate in zip(image_paths, results):
            print(is_inappropriate)
            print(f"{path}: {'Inappropriate' if is_inappropriate else 'Safe'}")
            
        # Check image with accompanying text
        result_with_text = detector.detect_image(
            "/home/ubuntu/xiaolong/jailbreakbench/unsafe.png",
            text="Check this image"
        )
        print("Flagged:", result_with_text.flagged)
        
    except Exception as e:
        print(f"Detection failed: {str(e)}")