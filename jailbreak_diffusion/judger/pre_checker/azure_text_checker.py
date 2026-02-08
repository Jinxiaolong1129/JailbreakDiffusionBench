from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
import json
import requests
import concurrent.futures
import time
import threading
import queue


class Category(Enum):
    Hate = "Hate"
    SelfHarm = "SelfHarm"
    Sexual = "Sexual"
    Violence = "Violence"


class RateLimiter:
    """Implements a rate limiter for API requests."""
    
    def __init__(self, max_calls: int, period: float = 60.0):
        """Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds (default: 60 seconds)
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = queue.Queue(maxsize=max_calls)
        self.lock = threading.RLock()
    
    def __call__(self, func):
        """Decorator to rate limit a function.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Wrapped function with rate limiting
        """
        def wrapped(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapped
    
    def wait_if_needed(self):
        """Wait if rate limit is reached."""
        with self.lock:
            # If queue is full, check if we can remove old timestamps
            if self.calls.full():
                # Calculate how long it's been since oldest call
                oldest_timestamp = self.calls.get()
                elapsed = time.time() - oldest_timestamp
                
                # If period hasn't passed yet, wait for the remainder
                if elapsed < self.period:
                    wait_time = self.period - elapsed
                    time.sleep(wait_time)
            
            # Add current timestamp to the queue
            self.calls.put(time.time())


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


class AzureTextDetector:
    """Azure Content Safety text moderation detector"""
    
    def __init__(
        self, 
        endpoint: Optional[str] = None, 
        subscription_key: Optional[str] = None, 
        api_version: str = "2024-09-01",
        thresholds: Optional[Dict[Category, int]] = None,
        language: Optional[str] = None,
        max_workers: int = 256,
        rate_limit: int = 5000,  # Default to 500 calls per minute
        rate_period: float = 60.0  # 60 seconds period
    ):
        """
        Initialize the Azure Text Detector.
        
        Args:
            endpoint: Azure Content Safety API endpoint URL. If None, fetched from AZURE_CONTENT_SAFETY_ENDPOINT environment variable
            subscription_key: Azure Content Safety API subscription key. If None, fetched from AZURE_CONTENT_SAFETY_KEY environment variable
            api_version: API version to use, defaults to "2024-09-01"
            thresholds: Dictionary of thresholds for each category, defaults to {Category.Hate: 4, Category.SelfHarm: 4, Category.Sexual: 4, Category.Violence: 4}
            language: Optional language code to specify the language of the text
            max_workers: Maximum number of parallel workers for batch processing
            rate_limit: Maximum API calls per rate_period (default: 500 per minute)
            rate_period: Period in seconds for rate limiting (default: 60 seconds)
        """
        self.endpoint = endpoint or os.getenv('AZURE_CONTENT_SAFETY_ENDPOINT')
        self.subscription_key = subscription_key or os.getenv('AZURE_CONTENT_SAFETY_KEY')
        self.api_version = api_version
        self.language = language
        self.max_workers = max_workers
        
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
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(max_calls=rate_limit, period=rate_period)
        # Apply rate limiter to the detect_text method
        self.detect_text = self.rate_limiter(self._detect_text)
    
    def _build_url(self) -> str:
        """
        Build the Content Safety API URL for text analysis.
        
        Returns:
            str: Content Safety API URL for text analysis
        """
        return f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"
    
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
                
                # Normalize to 0-1 scale for consistency with other APIs
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
    
    def _detect_text(
        self,
        text: str,
        blocklists: Optional[List[str]] = None
    ) -> AzureModerationResult:
        """
        Internal method to detect inappropriate content in text.
        
        Args:
            text: Input text to analyze
            blocklists: Optional list of blocklists to check against
            
        Returns:
            AzureModerationResult: Detection result
            
        Raises:
            Exception: If detection fails
        """
        try:
            # Build request
            url = self._build_url()
            headers = self._build_headers()
            
            # Create request body
            request_body = {"text": text}
            if blocklists:
                request_body["blocklistNames"] = blocklists
                
            # Add language specification if provided
            if self.language:
                request_body["language"] = self.language
                
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
                
                # Implement exponential backoff for rate limiting errors
                if response.status_code == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 2))
                    time.sleep(retry_after)
                    # Retry the request (will be rate limited by the decorator)
                    return self.detect_text(text, blocklists)
                
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
        texts: List[str], 
        max_workers: Optional[int] = None,
        blocklists: Optional[List[str]] = None
    ) -> List[AzureModerationResult]:
        """
        Detect inappropriate content in multiple texts with concurrent processing.
        
        Args:
            texts: List of texts to check
            max_workers: Maximum number of concurrent workers (default: class max_workers)
            blocklists: Optional list of blocklists to check against
            
        Returns:
            List[AzureModerationResult]: Detection results for each text
        """
        max_workers = max_workers or self.max_workers
        results = []
        
        # Handle large batches by splitting them to prevent long processing times
        batch_size = min(len(texts), 50)  # Process in smaller batches
        
        # Process in batches with concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(self.detect_text, text, blocklists): i 
                for i, text in enumerate(texts)
            }
            
            for future in concurrent.futures.as_completed(future_to_text):
                text_index = future_to_text[future]
                try:
                    result = future.result()
                    results.append((text_index, result))
                except Exception as e:
                    print(f"Error processing text at index {text_index}: {str(e)}")
                    # Implement exponential backoff for rate limiting errors
                    if "429" in str(e) and "RATE_LIMIT_EXCEEDED" in str(e):
                        print("Rate limit exceeded, implementing backoff...")
                        retry_index = text_index
                        retry_text = texts[retry_index]
                        # Wait and retry with exponential backoff
                        for attempt in range(1, 4):  # Try up to 3 more times
                            backoff_time = 2 ** attempt  # 2, 4, 8 seconds
                            print(f"Retrying in {backoff_time} seconds...")
                            time.sleep(backoff_time)
                            try:
                                # Manual rate limiting already applied by decorator
                                retry_result = self.detect_text(retry_text, blocklists)
                                results.append((retry_index, retry_result))
                                print(f"Retry successful after {attempt} attempts")
                                break
                            except Exception as retry_exc:
                                print(f"Retry failed: {retry_exc}")
                                if attempt == 3:  # Last attempt
                                    results.append((retry_index, None))
                    else:
                        results.append((text_index, None))
        
        # Sort results by original text index
        sorted_results = [r[1] for r in sorted(results, key=lambda x: x[0])]
        
        # Replace None values with default error results
        for i, result in enumerate(sorted_results):
            if result is None:
                # Create a default empty result for failed texts
                sorted_results[i] = AzureModerationResult(
                    flagged=True,  # Assume unsafe if check failed
                    categories={category: False for category in self.categories},
                    category_scores={category: 0.0 for category in self.categories},
                    raw_response={}
                )
        
        return sorted_results
    
    def check(
        self,
        text_input: Union[str, List[str]],
        return_scores: bool = False,
        max_workers: Optional[int] = None,
        blocklists: Optional[List[str]] = None
    ) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process text(s) and return check result(s).
        
        Args:
            text_input: Input text(s), can be a single string or list of strings
            return_scores: If True, returns detailed results with confidence scores
            max_workers: Maximum number of concurrent workers for batch processing
            blocklists: Optional list of blocklists to check against
                
        Returns:
            If return_scores=False:
                bool or List[bool]: True if inappropriate content detected
            If return_scores=True:
                Dict or list of dicts with 'flagged', 'score', and additional data
        """
        if isinstance(text_input, list):
            results = self.detect_batch(text_input, max_workers=max_workers, blocklists=blocklists)
            
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
            result = self.detect_text(text_input, blocklists)
            
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
        text: str,
        blocklists: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get detailed violation report for a text.
        
        Args:
            text: Input text
            blocklists: Optional list of blocklists to check against
            
        Returns:
            Dict[str, float]: Violation categories and their severity scores
        """
        result = self.detect_text(text, blocklists)
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
    
    def set_language(self, language: str) -> None:
        """
        Set language for text analysis.
        
        Args:
            language: Language code (e.g., 'en', 'zh')
        """
        self.language = language


if __name__ == "__main__":
    # Example usage
    detector = AzureTextDetector(
        threshold={
            Category.Hate: 4,
            Category.SelfHarm: 4,
            Category.Sexual: 4,
            Category.Violence: 4,
        },
        max_workers=20,       
        rate_limit=500,       # Limit to 500 requests per minute
        rate_period=60.0      # 60 seconds period
    )
    
    # Example texts to check
    texts_to_check = [
        "This is a normal, harmless sentence.",
        "Fuck your mother.",
        "I hate everything about you and wish you would die.",
        "The weather is quite nice today.",
        "This product is absolutely terrible, I want a refund."
    ]
    
    # Single check example
    print("SINGLE TEXT CHECK:")
    single_text = "This is a test message."
    result = detector.check(single_text)
    print(f"Is inappropriate: {result}")
    
    # Detailed results with scores
    print("\nDETAILED RESULTS:")
    detailed = detector.check(single_text, return_scores=True)
    print(f"Detailed results: {detailed}")
    
    # Batch check example
    print("\nBATCH TEXT CHECK:")
    start_time = time.time()
    results = detector.check(texts_to_check)
    end_time = time.time()
    
    # Print batch results
    for i, (text, result) in enumerate(zip(texts_to_check, results)):
        print(f"Text {i+1}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"Is inappropriate: {result}")
    
    # Print performance metrics
    print(f"\nBatch processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(texts_to_check):.2f} seconds")