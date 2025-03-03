from typing import Union, List, Dict, Optional
from google.cloud import language_v1
from concurrent.futures import ThreadPoolExecutor
import time

from base_checker import BaseChecker

class GoogleTextChecker(BaseChecker):
    """Content checker that uses Google's Text Moderation API with multi-threading support."""
    
    def __init__(self, 
                 threshold: float = 0.6,
                 language: Optional[str] = None,
                 max_workers: int = 10):
        """Initialize the Google Text Moderation checker.
        
        Args:
            threshold: Confidence threshold for any category to be considered inappropriate.
                       If any category's confidence exceeds this value, content is flagged.
            language: Optional language code (e.g., 'en', 'zh'). If None, API auto-detects.
            max_workers: Maximum number of parallel workers for batch processing
        """
        super().__init__()
        self.client = language_v1.LanguageServiceClient()
        self.threshold = threshold
        self.language = language
        self.max_workers = max_workers
    
    def moderate_text(self, text_content: str) -> language_v1.ModerateTextResponse:
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
        """Moderate multiple texts in parallel using ThreadPoolExecutor.
        
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
                    results.append((future_to_text[future], None))
        
        # Sort by original index and return just the results
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if text contains inappropriate content based on threshold.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        if isinstance(text, str):
            return self._check_single(text)
        else:
            # Use parallel processing for batch checks
            responses = self.moderate_texts(text)
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
    checker = GoogleTextChecker(threshold=0.6)  # Using the default threshold of 0.6
    
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
    result = checker(single_text)
    print(f"Is inappropriate: {result}")
    
    # Batch check example
    print("\nBATCH TEXT CHECK:")
    start_time = time.time()
    results = checker(texts_to_check)
    end_time = time.time()
    
    # Print batch results
    for i, (text, result) in enumerate(zip(texts_to_check, results)):
        print(f"Text {i+1}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"Is inappropriate: {result}")
    
    # Print performance metrics
    print(f"\nBatch processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(texts_to_check):.2f} seconds")
    
    # Get detailed results example
    print("\nDETAILED RESULTS EXAMPLE:")
    detailed_results = checker.get_detailed_results(texts_to_check[1])  # Get results for the offensive text
    for category, confidence in sorted(detailed_results.items(), key=lambda x: x[1], reverse=True):
        flag = "⚠️ " if confidence >= checker.threshold else "  "
        print(f"{flag}{category}: {confidence:.2f}")