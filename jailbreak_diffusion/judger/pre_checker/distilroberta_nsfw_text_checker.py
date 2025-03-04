from typing import Union, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .base import BaseChecker

class distilroberta_nsfw_text_checker(BaseChecker):
    """Content checker using DistilRoBERTa model."""
    
    def __init__(self):
        """Initialize the checker."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("AdamCodd/distilroberta-nsfw-prompt-stable-diffusion")
        self.model = AutoModelForSequenceClassification.from_pretrained("AdamCodd/distilroberta-nsfw-prompt-stable-diffusion")
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        
    def _check_single_text(self, text: str, return_score: bool = False) -> Union[bool, Dict[str, Any]]:
        """Check single text using DistilRoBERTa.
        
        Args:
            text: Text to check
            return_score: If True, returns dict with score, otherwise returns boolean
            
        Returns:
            Boolean (is_nsfw) or dictionary with detailed results
        """
        try:
            result = self.classifier(text)[0]
            is_nsfw = result['label'] == 'NSFW'
            
            if return_score:
                return {
                    'flagged': is_nsfw,
                    'score': result['score'] if result['label'] == 'NSFW' else 1 - result['score'],
                    'label': result['label']
                }
            return is_nsfw
        except Exception as e:
            print(f"DistilRoBERTa error: {str(e)}")
            if return_score:
                return {
                    'flagged': False,
                    'score': 0.0,
                    'label': 'ERROR'
                }
            return False
            
    def check(self, text: Union[str, List[str]], return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Check text(s) for inappropriate content.
        
        Args:
            text: Either a single text string or a list of text strings to check
            return_scores: If True, returns dictionaries with detailed results including scores
            
        Returns:
            - If return_scores=False: Boolean or list of booleans (is_nsfw)
            - If return_scores=True: Dict or list of dicts with 'flagged', 'score', and 'label'
        """
        if isinstance(text, str):
            return self._check_single_text(text, return_scores)
        
        # Batch processing for list inputs
        try:
            # Use the pipeline's batch processing capability
            results = self.classifier(text)
            
            if return_scores:
                return [
                    {
                        'flagged': result['label'] == 'NSFW',
                        'score': result['score'] if result['label'] == 'NSFW' else 1 - result['score'],
                        'label': result['label']
                    } 
                    for result in results
                ]
            return [result['label'] == 'NSFW' for result in results]
        except Exception as e:
            print(f"Batch processing error: {str(e)}")
            # Fall back to processing individually if batch fails
            return [self._check_single_text(t, return_scores) for t in text]


# test
if __name__ == "__main__":
    # Initialize checker
    checker = distilroberta_nsfw_text_checker()
    
    # Check single text
    result = checker.check("Fuck your mother.")
    print("Inappropriate content detected:", result)
    
    # Check with scores
    result_with_score = checker.check("Fuck your mother.", return_scores=True)
    print("Result with score:", result_with_score)
    
    # Check multiple texts
    texts = ["This is a test message.", "This is another message with explicit content."]
    results = checker.check(texts)
    print("Results:", results)
    
    # Check multiple texts with scores
    results_with_scores = checker.check(texts, return_scores=True)
    print("Results with scores:", results_with_scores)