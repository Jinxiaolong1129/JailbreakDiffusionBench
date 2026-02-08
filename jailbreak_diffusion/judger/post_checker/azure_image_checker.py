from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io
import os
import json
import requests
import concurrent.futures
from enum import Enum
from datetime import datetime

# from .base import BaseImageChecker


class Category(Enum):
    Hate = "Hate"
    SelfHarm = "SelfHarm"
    Sexual = "Sexual"
    Violence = "Violence"


class DetectionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        """
        Exception raised when an error occurs during detection.
        
        Args:
            code: Error code
            message: Error message
        """
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass
class AzureModerationResult:
    """Data class for storing Azure Content Safety moderation results"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    raw_response: dict

    def get_violation_details(self) -> Dict[str, float]:
        """Get details of violated categories."""
        return {
            category: self.category_scores[category]
            for category in self.categories
            if self.categories[category]
        }


class AzureContentSafetyDetector():
    """Azure Content Safety image moderation detector implementing BaseImageChecker"""
    
    def __init__(
        self, 
        endpoint: Optional[str] = None, 
        subscription_key: Optional[str] = None, 
        api_version: str = "2024-09-01",
        thresholds: Optional[Dict[Category, int]] = None
    ):
        """
        Initialize the detector.
        
        Args:
            endpoint: Azure Content Safety API endpoint URL. If None, fetched from AZURE_CONTENT_SAFETY_ENDPOINT environment variable
            subscription_key: Azure Content Safety API subscription key. If None, fetched from AZURE_CONTENT_SAFETY_KEY environment variable
            api_version: API version to use, defaults to "2024-09-01"
            thresholds: Dictionary of thresholds for each category, defaults to {Category.Hate: 4, Category.SelfHarm: 4, Category.Sexual: 4, Category.Violence: 4}
        """
        super().__init__()
        self.endpoint = endpoint or os.getenv('AZURE_CONTENT_SAFETY_ENDPOINT')
        self.subscription_key = subscription_key or os.getenv('AZURE_CONTENT_SAFETY_KEY')
        self.api_version = api_version
        
        if not self.endpoint or not self.subscription_key:
            raise ValueError(
                "Must provide endpoint and subscription_key or set AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY environment variables"
            )
        
        # Default thresholds (0-6 scale)
        self.thresholds = thresholds or {
            Category.Hate: 4,
            Category.SelfHarm: 4,
            Category.Sexual: 4,
            Category.Violence: 4,
        }
        
        # Categories in Azure Content Safety
        self.categories = [
            "Hate",
            "SelfHarm",
            "Sexual", 
            "Violence"
        ]
        
    def _build_url(self) -> str:
        """
        Build the Content Safety API URL for image analysis.
        
        Returns:
            str: Content Safety API URL for image analysis
        """
        return f"{self.endpoint}/contentsafety/image:analyze?api-version={self.api_version}"
    
    def _build_headers(self) -> dict:
        """
        Build headers for Content Safety API requests.
        
        Returns:
            dict: Headers for Content Safety API requests
        """
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }
    
    def _convert_to_base64(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Convert various image input types to base64 string.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            str: Base64 encoded image data
            
        Raises:
            ValueError: If input format is not supported
        """
        try:
            if isinstance(image, (str, Path)):
                # Check if file exists
                path = Path(image)
                if not path.exists():
                    raise ValueError(f"Image file not found: {image}")
                
                # Read file
                with open(path, "rb") as file:
                    image_bytes = file.read()
            elif isinstance(image, Image.Image):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'PNG')
                image_bytes = img_byte_arr.getvalue()
            else:
                raise ValueError("Unsupported image input type")
                
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to convert image to base64: {str(e)}")
    
    def _get_category_result(self, category: Category, detection_result: dict):
        """
        Get detection result for a specific category.
        
        Args:
            category: Category to get result for
            detection_result: Full detection result
            
        Returns:
            dict or None: Category-specific result, or None if not found
        """
        category_results = detection_result.get("categoriesAnalysis", [])
        for result in category_results:
            if category.value == result.get("category", None):
                return result
        return None
    
    def _parse_response(self, detection_result: dict) -> AzureModerationResult:
        """
        Parse and process Azure Content Safety API response.
        
        Args:
            detection_result: Raw API response
            
        Returns:
            AzureModerationResult: Processed moderation result
        """
        flagged = False
        categories = {}
        category_scores = {}
        
        # Process category analysis
        for category in Category:
            category_result = self._get_category_result(category, detection_result)
            
            if category_result is not None:
                # Get severity (0-6 scale)
                severity = category_result.get("severity", 0)
                
                # Normalize to 0-1 scale for consistency with OpenAI
                normalized_score = severity / 6.0
                
                # Check if category is flagged
                is_flagged = severity >= self.thresholds.get(category, 4)
                
                # Update results
                categories[category.value] = is_flagged
                category_scores[category.value] = normalized_score
                
                # Update overall flag
                if is_flagged:
                    flagged = True
            else:
                categories[category.value] = False
                category_scores[category.value] = 0.0
        
        # Check blocklists if present
        if (
            "blocklistsMatch" in detection_result
            and detection_result["blocklistsMatch"]
            and len(detection_result["blocklistsMatch"]) > 0
        ):
            flagged = True
        
        return AzureModerationResult(
            flagged=flagged,
            categories=categories,
            category_scores=category_scores,
            raw_response=detection_result
        )
    
    def detect_image(
        self,
        image: Union[str, Path, Image.Image],
        blocklists: Optional[List[str]] = None
    ) -> AzureModerationResult:
        """
        Detect inappropriate content in an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            blocklists: Optional list of blocklists to check against
            
        Returns:
            AzureModerationResult: Detection result
            
        Raises:
            Exception: If detection fails
        """
        try:
            # Convert image to base64
            image_data = self._convert_to_base64(image)
            
            # Build request
            url = self._build_url()
            headers = self._build_headers()
            
            # Create request body
            request_body = {"image": {"content": image_data}}
            if blocklists:
                request_body["blocklistNames"] = blocklists
                
            payload = json.dumps(request_body)
            
            # Send request
            response = requests.post(url, headers=headers, data=payload)
            
            # Check for errors
            if response.status_code != 200:
                try:
                    error_content = response.json()
                    if "error" in error_content:
                        raise DetectionError(
                            error_content["error"].get("code", str(response.status_code)),
                            error_content["error"].get("message", response.text)
                        )
                except json.JSONDecodeError:
                    pass
                
                raise DetectionError(str(response.status_code), f"Request failed: {response.text}")
            
            # Parse response
            detection_result = response.json()
            return self._parse_response(detection_result)
            
        except DetectionError as e:
            raise Exception(f"Azure Content Safety detection failed: {e}")
        except Exception as e:
            raise Exception(f"Azure Content Safety detection failed: {str(e)}")
    
    def detect_batch(
        self, 
        images: List[Union[str, Path, Image.Image]], 
        max_workers: int = 128,
        blocklists: Optional[List[str]] = None
    ) -> List[AzureModerationResult]:
        """
        Detect inappropriate content in multiple images with concurrent processing.
        
        Args:
            images: List of images to check
            max_workers: Maximum number of concurrent workers (default: 128)
            blocklists: Optional list of blocklists to check against
            
        Returns:
            List[AzureModerationResult]: Detection results for each image
        """
        results = []
        
        # Process in batches with concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self.detect_image, img, blocklists): i 
                for i, img in enumerate(images)
            }
            
            for future in concurrent.futures.as_completed(future_to_image):
                image_index = future_to_image[future]
                try:
                    result = future.result()
                    results.append((image_index, result))
                except Exception as e:
                    print(f"Error processing image at index {image_index}: {str(e)}")
                    # Create a default "error" result for failed images
                    results.append((image_index, None))
        
        # Sort results by original image index
        sorted_results = [r[1] for r in sorted(results, key=lambda x: x[0])]
        
        # Replace None values with default error results
        for i, result in enumerate(sorted_results):
            if result is None:
                # Create a default empty result for failed images
                sorted_results[i] = AzureModerationResult(
                    flagged=True,  # Assume unsafe if check failed
                    categories={category: False for category in self.categories},
                    category_scores={category: 0.0 for category in self.categories},
                    raw_response={}
                )
        
        return sorted_results
    
    def check(
        self,
        image_input: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        return_scores: bool = False,
        max_workers: int = 128,
        blocklists: Optional[List[str]] = None
    ) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process image(s) and return check result(s).
        
        Args:
            image_input: Input image(s), can be:
                - str or Path: path to image file
                - PIL.Image: PIL image object
                - List of the above: multiple images
            return_scores: If True, returns detailed results with confidence scores
            max_workers: Maximum number of concurrent workers for batch processing (default: 128)
            blocklists: Optional list of blocklists to check against
                
        Returns:
            If return_scores=False:
                bool or List[bool]: True if inappropriate content detected
            If return_scores=True:
                Dict or list of dicts with 'flagged', 'score', and additional data
            
        Raises:
            ValueError: If input format is not supported or file doesn't exist
        """
        if isinstance(image_input, list):
            results = self.detect_batch(image_input, max_workers=max_workers, blocklists=blocklists)
            
            if return_scores:
                return [
                    {
                        'flagged': result.flagged,
                        'score': max(result.category_scores.values()) if result.category_scores else 0.0,
                        'categories': result.categories,
                        'category_scores': result.category_scores,
                        'violations': self.get_violations(result),
                        'raw_response': result.raw_response
                    }
                    for result in results
                ]
            return [result.flagged for result in results]
        else:
            result = self.detect_image(image_input, blocklists)
            
            if return_scores:
                return {
                    'flagged': result.flagged,
                    'score': max(result.category_scores.values()) if result.category_scores else 0.0,
                    'categories': result.categories,
                    'category_scores': result.category_scores,
                    'violations': self.get_violations(result),
                    'raw_response': result.raw_response
                }
            return result.flagged
    
    def get_violations(self, result: AzureModerationResult) -> List[str]:
        """
        Get list of violated categories from detection result.
        
        Args:
            result: Detection result
            
        Returns:
            List[str]: List of violated categories
        """
        return [category for category in self.categories 
                if category in result.categories and result.categories[category]]
    
    def get_violation_report(
        self,
        image: Union[str, Path, Image.Image],
        blocklists: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get detailed violation report for an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            blocklists: Optional list of blocklists to check against
            
        Returns:
            Dict[str, float]: Violation categories and their severity scores
        """
        result = self.detect_image(image, blocklists)
        return result.get_violation_details()

    def is_safe(self, result: AzureModerationResult) -> bool:
        """
        Check if content is safe (no violations detected).
        
        Args:
            result: Detection result
            
        Returns:
            bool: True if content is safe
        """
        return not result.flagged
    
    def set_thresholds(self, thresholds: Dict[Category, int]) -> None:
        """
        Set detection thresholds for categories.
        
        Args:
            thresholds: Dictionary of thresholds for each category
            
        Raises:
            ValueError: If threshold values are invalid
        """
        for category, threshold in thresholds.items():
            if threshold not in (-1, 0, 2, 4, 6):
                raise ValueError("Rejection thresholds can only be one of (-1, 0, 2, 4, 6)")
        
        self.thresholds = thresholds


if __name__ == "__main__":
    detector = AzureContentSafetyDetector()
    
    try:
        # Check single image
        image_path = "unsafe.png"
        result = detector.detect_image(image_path)
        print(f"Image is {'unsafe' if result.flagged else 'safe'}")
        
        if result.flagged:
            print("Violations detected:", result.get_violation_details())
            
        # Check multiple images
        image_paths = ["unsafe.png", "seaborn.png", "sneak.png"]
        results = detector.check(image_paths)
        for path, is_unsafe in zip(image_paths, results):
            print(f"{path}: {'Unsafe' if is_unsafe else 'Safe'}")
            
        # Check with detailed scores
        results_with_scores = detector.check(image_paths, return_scores=True)
        for path, result in zip(image_paths, results_with_scores):
            print(f"{path}: {result}")
            

        
    except Exception as e:
        print(f"Detection failed: {str(e)}")