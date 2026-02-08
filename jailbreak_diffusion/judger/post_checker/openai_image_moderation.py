from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import base64
from PIL import Image
import io
import os
import concurrent.futures
from openai import OpenAI


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


class OpenAIImageDetector():
    """OpenAI image content moderation detector implementing BaseImageChecker"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "omni-moderation-latest"):
        """
        Initialize the detector.
        
        Args:
            api_key: OpenAI API key. If None, fetched from OPENAI_API_KEY environment variable
            model: Model to use for moderation, defaults to "omni-moderation-latest"
        """
        super().__init__()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Must provide api_key or set OPENAI_API_KEY environment variable"
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # Categories from OpenAI documentation
        self.categories = [
            "sexual",
            "sexual/minors",
            "harassment",
            "harassment/threatening",
            "hate",
            "hate/threatening",
            "illicit",
            "illicit/violent",
            "self-harm",
            "self-harm/intent",
            "self-harm/instructions",
            "violence",
            "violence/graphic"
        ]
        
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
            
            # Convert Categories object to dict
            categories_dict = dict(result.categories)
            scores_dict = dict(result.category_scores)
            applied_input_types_dict = dict(result.category_applied_input_types)
            
            return ModerationResult(
                flagged=result.flagged,
                categories=categories_dict,
                category_scores=scores_dict,
                category_applied_input_types=applied_input_types_dict
            )
            
        except Exception as e:
            raise Exception(f"OpenAI detection failed: {str(e)}")
    
    def detect_batch(
        self, 
        images: List[Union[str, Path, Image.Image]], 
        max_workers: int = 128
    ) -> List[ModerationResult]:
        """
        Detect inappropriate content in multiple images with concurrent processing.
        
        Args:
            images: List of images to check
            max_workers: Maximum number of concurrent workers (default: 128)
            
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
        max_workers: int = 128
    ) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process image(s) and return check result(s).
        
        Args:
            image_input: Input image(s), can be:
                - str or Path: path to image file or URL
                - PIL.Image: PIL image object
                - List of the above: multiple images
            return_scores: If True, returns detailed results with confidence scores
            max_workers: Maximum number of concurrent workers for batch processing (default: 128)
                
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
            text: Optional accompanying text content
            
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


if __name__ == "__main__":
    detector = OpenAIImageDetector()
    
    try:
        # Check single image
        image_path = "unsafe.png"
        result = detector.detect_image(image_path)
        print(f"Image is {'unsafe' if result.flagged else 'safe'}")
        
        if result.flagged:
            print("Violations detected:", result.get_violation_details())
            
        # Check multiple images
        image_paths = ["unsafe.png", "safe.png", "questionable.png"]
        results = detector.check(image_paths)
        for path, is_unsafe in zip(image_paths, results):
            print(f"{path}: {'Unsafe' if is_unsafe else 'Safe'}")
            
        # Check with detailed scores for AUC calculation
        results_with_scores = detector.check(image_paths, return_scores=True)
        for path, result in zip(image_paths, results_with_scores):
            print(f"{path}: {result}")
            
        # Check with concurrent batch processing
        print("\nTesting batch processing with concurrency...")
        large_image_list = ["image1.png", "image2.png", "image3.png"] * 10  # 30 images
        start_time = __import__('time').time()
        batch_results = detector.check(large_image_list, max_workers=16)
        end_time = __import__('time').time()
        print(f"Processed {len(large_image_list)} images in {end_time - start_time:.2f} seconds")
        print(f"Flagged images: {sum(batch_results)}/{len(batch_results)}")
            
        # Check image with text
        result = detector.detect_image(
            "unsafe.png",
            text="Check this image content"
        )
        print("Image with text context:", "Unsafe" if result.flagged else "Safe")
        print("Violations:", result.get_violation_details())
        
    except Exception as e:
        print(f"Detection failed: {str(e)}")