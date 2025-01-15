from typing import Dict, Union, Optional, List
from pathlib import Path
from dataclasses import dataclass
import os
from PIL import Image
import io

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeImageOptions,
    ImageData,
    ImageCategory,
    ImageAnalysisResult,
    CategoryAnalysis
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from .base import BaseImageChecker

@dataclass
class ContentSafetyResult:
    """Data class for storing content safety detection results."""
    hate_severity: float
    self_harm_severity: float
    sexual_severity: float
    violence_severity: float
    
    def is_safe(self, threshold: float = 0.5) -> bool:
        """
        Determine if the content is safe.
        
        Args:
            threshold: Safety threshold, defaults to 0.5
            
        Returns:
            bool: True if all category severities are below threshold
        """
        return all(
            severity < threshold
            for severity in [
                self.hate_severity,
                self.self_harm_severity,
                self.sexual_severity,
                self.violence_severity
            ]
        )
    
    def get_violations(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Get categories that exceed the threshold.
        
        Args:
            threshold: Violation threshold, defaults to 0.5
            
        Returns:
            Dict[str, float]: Violation categories and their severity scores
        """
        violations = {}
        if self.hate_severity >= threshold:
            violations['hate'] = self.hate_severity
        if self.self_harm_severity >= threshold:
            violations['self_harm'] = self.self_harm_severity
        if self.sexual_severity >= threshold:
            violations['sexual'] = self.sexual_severity
        if self.violence_severity >= threshold:
            violations['violence'] = self.violence_severity
        return violations

class AzureImageDetector(BaseImageChecker):
    """Azure Content Safety image detector implementing BaseImageChecker."""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize the detector.
        
        Args:
            endpoint: Azure Content Safety endpoint. If None, fetched from environment
            key: Azure Content Safety key. If None, fetched from environment
            threshold: Safety threshold for detection, defaults to 0.5
        """
        super().__init__()
        self.endpoint = endpoint or os.environ.get('CONTENT_SAFETY_ENDPOINT')
        self.key = key or os.environ.get('CONTENT_SAFETY_KEY')
        self.threshold = threshold
        
        if not self.endpoint or not self.key:
            raise ValueError(
                "Must provide endpoint and key, or set environment variables "
                "CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_KEY"
            )
            
        self.client = ContentSafetyClient(
            self.endpoint,
            AzureKeyCredential(self.key)
        )
    
    def _get_category_severity(
        self,
        categories_analysis: List[CategoryAnalysis],
        category: ImageCategory
    ) -> float:
        """
        Get severity score for a specific category.
        
        Args:
            categories_analysis: List of category analysis results
            category: Category to look up
            
        Returns:
            float: Category severity score
        """
        try:
            result = next(
                item for item in categories_analysis 
                if item.category == category
            )
            return result.severity
        except StopIteration:
            return 0.0
    
    def _convert_to_bytes(self, image: Union[str, Path, Image.Image]) -> bytes:
        """
        Convert various image input types to bytes.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            bytes: Image content as bytes
        """
        if isinstance(image, (str, Path)):
            with open(image, "rb") as file:
                return file.read()
        elif isinstance(image, Image.Image):
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'PNG')
            return img_byte_arr.getvalue()
        else:
            raise ValueError("Unsupported image input type")
    
    def detect_image(self, image: Union[str, Path, Image.Image]) -> ContentSafetyResult:
        """
        Detect content safety issues in a single image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            ContentSafetyResult: Detection result
            
        Raises:
            Exception: If detection fails
        """
        try:
            content = self._convert_to_bytes(image)
            request = AnalyzeImageOptions(
                image=ImageData(content=content)
            )
            response = self.client.analyze_image(request)
            
            return ContentSafetyResult(
                hate_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.HATE
                ),
                self_harm_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.SELF_HARM
                ),
                sexual_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.SEXUAL
                ),
                violence_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.VIOLENCE
                )
            )
            
        except HttpResponseError as e:
            error_message = "Detection failed."
            if e.error:
                error_message += f"\nError code: {e.error.code}"
                error_message += f"\nError message: {e.error.message}"
            raise Exception(error_message) from e
        except Exception as e:
            raise Exception(f"Error during detection: {str(e)}") from e
    
    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        
        Args:
            image_input: Input image(s), can be:
                - str or Path: path to image file
                - PIL.Image: PIL image object
                - List of the above: multiple images
                
        Returns:
            bool or List[bool]: True if inappropriate content detected
            
        Raises:
            ValueError: If input format is not supported or file doesn't exist
        """
        if isinstance(image_input, list):
            return [not self.detect_image(img).is_safe(self.threshold) for img in image_input]
        else:
            return not self.detect_image(image_input).is_safe(self.threshold)

    def get_violation_report(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict[str, float]:
        """
        Get detailed violation report for an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            Dict[str, float]: Violation categories and their severity scores
        """
        result = self.detect_image(image)
        return result.get_violations(self.threshold)

# Usage example
if __name__ == "__main__":
    # Initialize detector (will read credentials from environment variables)
    detector = AzureImageDetector(threshold=0.4)
    
    # Check single image
    try:
        # Using path
        result = detector("path/to/image.jpg")
        if result:
            print("Inappropriate content detected")
            # Get violation details
            violations = detector.get_violation_report("path/to/image.jpg")
            print("Violations detected:", violations)
        else:
            print("Image is safe")
            
        # Check multiple images
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        results = detector(image_paths)
        for path, is_inappropriate in zip(image_paths, results):
            print(f"{path}: {'Inappropriate' if is_inappropriate else 'Safe'}")
            
    except Exception as e:
        print(f"Detection failed: {str(e)}")