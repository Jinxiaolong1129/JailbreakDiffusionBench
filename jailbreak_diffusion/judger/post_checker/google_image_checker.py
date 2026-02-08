
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io
import os
import concurrent.futures
from google.cloud import vision
from google.oauth2 import service_account

# from .base import BaseImageChecker

@dataclass
class ModerationResult:
    """Data class for storing moderation results"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Dict[str, List[str]]

    def get_violation_details(self) -> Dict[str, float]:
        """Get details of violated categories."""
        return {
            category: self.category_scores[category]
            for category in self.categories
            if self.categories[category]
        }


class GoogleImageDetector():
    """Google Cloud Vision image content moderation detector implementing BaseImageChecker"""
    
    def __init__(self, credentials_path: Optional[str] = None, likelihood_threshold: int = 4):
        """
        Initialize the detector.
        
        Args:
            credentials_path: Path to Google credentials JSON file. 
                If None, fetched from GOOGLE_APPLICATION_CREDENTIALS environment variable
            likelihood_threshold: Threshold for likelihood score (2=UNLIKELY, 3=POSSIBLE, 4=LIKELY, 5=VERY_LIKELY)
                Default is 4 (LIKELY)
        """
        super().__init__()
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Initialize client with credentials if provided
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
            self.client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Try to use default credentials
            try:
                self.client = vision.ImageAnnotatorClient()
            except Exception as e:
                raise ValueError(
                    "Failed to initialize Vision client. Must provide credentials_path or "
                    "set GOOGLE_APPLICATION_CREDENTIALS environment variable. Error: " + str(e)
                )
        
        # Threshold for flagging content (4=LIKELY, 5=VERY_LIKELY)
        self.likelihood_threshold = likelihood_threshold
        
        # Likelihood name mapping
        self.likelihood_name = [
            "UNKNOWN",
            "VERY_UNLIKELY",
            "UNLIKELY",
            "POSSIBLE",
            "LIKELY",
            "VERY_LIKELY",
        ]
        
        # Categories from Google Vision SafeSearch
        self.categories = [
            "adult",
            "spoof",
            "medical",
            "violence",
            "racy"
        ]

    def _normalize_score(self, likelihood_score: int) -> float:
        """
        Convert Google's discrete likelihood value to a normalized score [0,1].
        
        Args:
            likelihood_score: Integer likelihood score (0-5)
            
        Returns:
            float: Normalized score between 0 and 1
        """
        # Convert to float in range [0,1]
        if likelihood_score is None or likelihood_score == 0:  # UNKNOWN
            return 0.0
        return (likelihood_score - 1) / 4.0  # Scale 1-5 to 0-1
        
    def _load_image(self, image: Union[str, Path, Image.Image]) -> vision.Image:
        """
        Load image from various inputs into Google Vision format.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            vision.Image: Google Vision image object
            
        Raises:
            ValueError: If input format is not supported
        """
        try:
            if isinstance(image, (str, Path)):
                # Handle URLs
                if str(image).startswith(('http://', 'https://')):
                    return vision.Image(source=vision.ImageSource(image_uri=str(image)))
                # Otherwise read file
                with open(image, "rb") as file:
                    content = file.read()
                return vision.Image(content=content)
            elif isinstance(image, Image.Image):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format or 'PNG')
                content = img_byte_arr.getvalue()
                return vision.Image(content=content)
            else:
                raise ValueError("Unsupported image input type")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def detect_image(
        self,
        image: Union[str, Path, Image.Image],
        text: Optional[str] = None
    ) -> ModerationResult:
        """
        Detect inappropriate content in an image using Google Cloud Vision.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            text: Optional accompanying text content (Not used in Google implementation)
            
        Returns:
            ModerationResult: Detection result instance
            
        Raises:
            Exception: If detection fails
        """
        try:
            # Load the image
            vision_image = self._load_image(image)
            
            # Call API
            response = self.client.safe_search_detection(image=vision_image)
            safe = response.safe_search_annotation
            
            # Check if we have an error
            if response.error.message:
                raise Exception(
                    f"Google Vision API error: {response.error.message}"
                )
            
            # Get likelihood scores for each category
            scores = {
                "adult": safe.adult,
                "spoof": safe.spoof,
                "medical": safe.medical,
                "violence": safe.violence,
                "racy": safe.racy
            }
            
            # Convert to normalized scores (0-1)
            normalized_scores = {
                category: self._normalize_score(score) 
                for category, score in scores.items()
            }
            
            # Determine if each category is flagged based on threshold
            flagged_categories = {
                category: score >= self.likelihood_threshold 
                for category, score in scores.items()
            }
            
            # Determine if the image is flagged overall
            is_flagged = any(flagged_categories.values())
            
            # Create applied input types dictionary (for compatibility with OpenAI)
            # In Google's case, all results are from the image
            applied_input_types = {
                category: ["image"] for category in self.categories if flagged_categories[category]
            }
            
            return ModerationResult(
                flagged=is_flagged,
                categories=flagged_categories,
                category_scores=normalized_scores,
                category_applied_input_types=applied_input_types
            )
            
        except Exception as e:
            raise Exception(f"Google Vision detection failed: {str(e)}")
    
    def detect_batch(
        self, 
        images: List[Union[str, Path, Image.Image]], 
        max_workers: int = 32
    ) -> List[ModerationResult]:
        """
        Detect inappropriate content in multiple images with concurrent processing.
        
        Args:
            images: List of images to check
            max_workers: Maximum number of concurrent workers (default: 32)
            
        Returns:
            List[ModerationResult]: Detection results for each image
        """
        results = []
        
        # Process in batches with concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {executor.submit(self.detect_image, img): i for i, img in enumerate(images)}
            
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
                sorted_results[i] = ModerationResult(
                    flagged=True,  # Assume unsafe if check failed
                    categories={category: False for category in self.categories},
                    category_scores={category: 0.0 for category in self.categories},
                    category_applied_input_types={}
                )
        
        return sorted_results
    
    def check(
        self,
        image_input: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        return_scores: bool = False,
        max_workers: int = 32
    ) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process image(s) and return check result(s).
        
        Args:
            image_input: Input image(s), can be:
                - str or Path: path to image file or URL
                - PIL.Image: PIL image object
                - List of the above: multiple images
            return_scores: If True, returns detailed results with confidence scores
            max_workers: Maximum number of concurrent workers for batch processing (default: 32)
                
        Returns:
            If return_scores=False:
                bool or List[bool]: True if inappropriate content detected
            If return_scores=True:
                Dict or list of dicts with 'flagged', 'score', and additional data
            
        Raises:
            ValueError: If input format is not supported or file doesn't exist
        """
        if isinstance(image_input, list):
            results = self.detect_batch(image_input, max_workers=max_workers)
            
            if return_scores:
                return [
                    {
                        'flagged': result.flagged,
                        'score': max(result.category_scores.values()) if result.category_scores else 0.0,
                        'categories': result.categories,
                        'category_scores': result.category_scores,
                        'violations': self.get_violations(result)
                    }
                    for result in results
                ]
            return [result.flagged for result in results]
        else:
            result = self.detect_image(image_input)
            
            if return_scores:
                return {
                    'flagged': result.flagged,
                    'score': max(result.category_scores.values()) if result.category_scores else 0.0,
                    'categories': result.categories,
                    'category_scores': result.category_scores,
                    'violations': self.get_violations(result)
                }
            return result.flagged
    
    def get_violations(self, result: ModerationResult) -> List[str]:
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
        text: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get detailed violation report for an image.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            text: Optional accompanying text content (not used in Google implementation)
            
        Returns:
            Dict[str, float]: Violation categories and their severity scores
        """
        result = self.detect_image(image, text)
        return result.get_violation_details()

    def is_safe(self, result: ModerationResult) -> bool:
        """
        Check if content is safe (no violations detected).
        
        Args:
            result: Detection result
            
        Returns:
            bool: True if content is safe
        """
        return not result.flagged

    def get_raw_likelihood_values(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict[str, str]:
        """
        Get raw likelihood values (text representation) for categories.
        
        Args:
            image: Input image (path string, Path object, or PIL Image)
            
        Returns:
            Dict[str, str]: Categories and their likelihood values as text
        """
        try:
            # Load the image
            vision_image = self._load_image(image)
            
            # Call API
            response = self.client.safe_search_detection(image=vision_image)
            safe = response.safe_search_annotation
            
            # Get likelihood names for each category
            likelihood_values = {
                "adult": self.likelihood_name[safe.adult],
                "spoof": self.likelihood_name[safe.spoof],
                "medical": self.likelihood_name[safe.medical],
                "violence": self.likelihood_name[safe.violence],
                "racy": self.likelihood_name[safe.racy]
            }
            
            return likelihood_values
            
        except Exception as e:
            raise Exception(f"Google Vision detection failed: {str(e)}")


if __name__ == "__main__":
    # Example usage
    detector = GoogleImageDetector()
    
    try:
        # Check single image
        image_path = "unsafe.png"
        result = detector.detect_image(image_path)
        print(f"Image is {'unsafe' if result.flagged else 'safe'}")
        
        if result.flagged:
            print("Violations detected:", result.get_violation_details())
            
        # Show raw likelihood values
        likelihood_values = detector.get_raw_likelihood_values(image_path)
        print("Raw likelihood values:")
        for category, likelihood in likelihood_values.items():
            print(f"{category}: {likelihood}")
            
        # Check multiple images
        image_paths = ["unsafe.png", "sneak.png", "seaborn.png"]
        results = detector.check(image_paths)
        for path, is_unsafe in zip(image_paths, results):
            print(f"{path}: {'Unsafe' if is_unsafe else 'Safe'}")
            
        # Check with detailed scores for AUC calculation
        results_with_scores = detector.check(image_paths, return_scores=True)
        for path, result in zip(image_paths, results_with_scores):
            print(f"{path}: {result}")
            
            
    except Exception as e:
        print(f"Detection failed: {str(e)}")