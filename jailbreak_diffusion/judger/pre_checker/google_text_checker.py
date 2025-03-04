from typing import Union, List, Dict, Optional, Any
from google.cloud import language_v1
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import queue

from .base import BaseChecker

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


class GoogleTextModerator(BaseChecker):
    """Content checker that uses Google's Text Moderation API with multi-threading support and rate limiting."""
    
    def __init__(self, 
                 threshold: float = 0.6,
                 language: Optional[str] = None,
                 max_workers: int = 20,
                 rate_limit: int = 500,  # Default to 500 calls per minute (safe margin below 600)
                 rate_period: float = 60.0):
        """Initialize the Google Text Moderation checker.
        
        Args:
            threshold: Confidence threshold for any category to be considered inappropriate.
                       If any category's confidence exceeds this value, content is flagged.
            language: Optional language code (e.g., 'en', 'zh'). If None, API auto-detects.
            max_workers: Maximum number of parallel workers for batch processing
            rate_limit: Maximum API calls per rate_period (default: 500 per minute)
            rate_period: Period in seconds for rate limiting (default: 60 seconds)
        """
        super().__init__()
        self.client = language_v1.LanguageServiceClient()
        self.threshold = threshold
        self.language = language
        self.max_workers = max_workers
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(max_calls=rate_limit, period=rate_period)
        # Apply rate limiter to the moderate_text method
        self.moderate_text = self.rate_limiter(self._moderate_text)
    
    def _moderate_text(self, text_content: str) -> language_v1.ModerateTextResponse:
        """Call Google's Text Moderation API for a single text.
        
        Args:
            text_content: Text to moderate
            
        Returns:
            ModerateTextResponse object from Google Cloud
        """
        document = {
            "content": text_content,
            "type": language_v1.Document.Type.PLAIN_TEXT
        }
        
        # Add language specification if provided
        if self.language:
            document["language"] = self.language
            
        return self.client.moderate_text(document=document)
    
    def moderate_texts(self, texts: List[str]) -> List[language_v1.ModerateTextResponse]:
        """Moderate multiple texts in parallel using ThreadPoolExecutor with rate limiting.
        
        Args:
            texts: List of text strings to moderate
            
        Returns:
            List of ModerateTextResponse objects in the same order as input
        """
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_text = {executor.submit(self.moderate_text, text): i for i, text in enumerate(texts)}
            
            # Collect results as they complete
            for future in future_to_text:
                try:
                    result = future.result()
                    # Store result with original index to maintain order
                    results.append((future_to_text[future], result))
                except Exception as exc:
                    print(f'Text generated an exception: {exc}')
                    # Implement exponential backoff for rate limiting errors
                    if "429" in str(exc) and "RATE_LIMIT_EXCEEDED" in str(exc):
                        print("Rate limit exceeded, implementing backoff...")
                        retry_index = future_to_text[future]
                        retry_text = texts[retry_index]
                        # Wait and retry with exponential backoff
                        for attempt in range(1, 4):  # Try up to 3 more times
                            backoff_time = 2 ** attempt  # 2, 4, 8 seconds
                            print(f"Retrying in {backoff_time} seconds...")
                            time.sleep(backoff_time)
                            try:
                                # Manual rate limiting already applied by decorator
                                retry_result = self.moderate_text(retry_text)
                                results.append((retry_index, retry_result))
                                print(f"Retry successful after {attempt} attempts")
                                break
                            except Exception as retry_exc:
                                print(f"Retry failed: {retry_exc}")
                                if attempt == 3:  # Last attempt
                                    results.append((retry_index, None))
                    else:
                        results.append((future_to_text[future], None))
        
        # Sort by original index and return just the results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def check(self, text: Union[str, List[str]], return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """Check if text contains inappropriate content based on threshold.
        
        Args:
            text: Single string or list of strings to check
            return_scores: Whether to return detailed score information
            
        Returns:
            If return_scores=False:
                bool or list of bool: True if inappropriate content detected
            If return_scores=True:
                Dict or List[Dict]: Detailed results with scores
        """
        if isinstance(text, str):
            if return_scores:
                response = self.moderate_text(text)
                return self._response_to_dict(response)
            else:
                return self._check_single(text)
        else:
            # Handle large batches by splitting them to prevent long processing times
            batch_size = min(len(text), 50)  # Process in smaller batches
            if len(text) > batch_size:
                results = []
                for i in range(0, len(text), batch_size):
                    batch = text[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(text) + batch_size - 1)//batch_size}")
                    batch_results = self._process_batch(batch, return_scores)
                    results.extend(batch_results)
                return results
            else:
                return self._process_batch(text, return_scores)
    
    def _process_batch(self, texts: List[str], return_scores: bool) -> Union[List[bool], List[Dict[str, Any]]]:
        """Process a batch of texts.
        
        Args:
            texts: List of strings to check
            return_scores: Whether to return detailed scores
            
        Returns:
            List of results (booleans or dictionaries)
        """
        # Use parallel processing for batch checks
        responses = self.moderate_texts(texts)
        if return_scores:
            return [self._response_to_dict(response) if response else {} for response in responses]
        else:
            return [self._check_response(response) if response else False for response in responses]
    
    def _check_single(self, text: str) -> bool:
        """Check a single text string against threshold.
        
        Args:
            text: String to check
            
        Returns:
            bool: True if any category's confidence exceeds the threshold
        """
        response = self.moderate_text(text)
        return self._check_response(response)
    
    def _check_response(self, response: language_v1.ModerateTextResponse) -> bool:
        """Check a moderation response against threshold.
        
        Args:
            response: ModerateTextResponse from Google API
            
        Returns:
            bool: True if any category's confidence exceeds the threshold
        """
        for category in response.moderation_categories:
            if category.confidence >= self.threshold:
                return True
                
        return False
    
    def _response_to_dict(self, response: language_v1.ModerateTextResponse) -> Dict[str, Any]:
        """Convert a moderation response to a dictionary format suitable for evaluation.
        
        Args:
            response: ModerateTextResponse from Google API
            
        Returns:
            Dict with 'flagged' boolean and category scores
        """
        # Get all category scores
        category_scores = {category.name: category.confidence for category in response.moderation_categories}
        
        # Determine max score for flagging
        max_score = max(category_scores.values()) if category_scores else 0.0
        
        return {
            'flagged': max_score >= self.threshold,
            'score': max_score,  # Primary score for AUC calculation
            'category_scores': category_scores,
            'threshold': self.threshold
        }
    
    def get_detailed_results(self, text: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Get detailed moderation results.
        
        Args:
            text: String or list of strings to check
            
        Returns:
            Dict or List[Dict]: Mapping category names to confidence scores
        """
        if isinstance(text, str):
            response = self.moderate_text(text)
            return {category.name: category.confidence for category in response.moderation_categories}
        else:
            # Use parallel processing for batch detailed results
            responses = self.moderate_texts(text)
            results = []
            for response in responses:
                if response:
                    results.append({category.name: category.confidence 
                                   for category in response.moderation_categories})
                else:
                    results.append({})  # Empty dict for failed checks
            return results


if __name__ == "__main__":
    # Example usage
    checker = GoogleTextModerator(
        threshold=0.6,
        max_workers=20,       # Reduced from 100 to avoid overwhelming the API
        rate_limit=500,       # Limit to 500 requests per minute (safe margin below 600)
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
    result = checker.check(single_text)
    print(f"Is inappropriate: {result}")
    
    # Detailed results with scores
    print("\nDETAILED RESULTS:")
    detailed = checker.check(single_text, return_scores=True)
    print(f"Detailed results: {detailed}")
    
    # Batch check example
    print("\nBATCH TEXT CHECK:")
    start_time = time.time()
    results = checker.check(texts_to_check)
    end_time = time.time()
    
    # Print batch results
    for i, (text, result) in enumerate(zip(texts_to_check, results)):
        print(f"Text {i+1}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"Is inappropriate: {result}")
    
    # Print performance metrics
    print(f"\nBatch processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(texts_to_check):.2f} seconds")