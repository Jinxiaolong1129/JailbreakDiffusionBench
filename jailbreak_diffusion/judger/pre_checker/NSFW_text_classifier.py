from typing import Union, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .base import BaseChecker


# True: NSFW
# False: Safe

class NSFW_text_classifier_Checker(BaseChecker):
    """Checker that uses a pre-trained model to detect inappropriate content."""
    def __init__(self):
        """Initialize with NSFW classifier model."""
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.classifier = None

    def _ensure_model_loaded(self):
        """Lazy loading of the model."""
        if self.classifier is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "michellejieli/NSFW_text_classifier"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "michellejieli/NSFW_text_classifier"
            )
            self.classifier = pipeline(
                "sentiment-analysis",
                tokenizer=self.tokenizer,
                model=self.model
            )

    def _check_single_text(self, text: str, return_score: bool = False) -> Union[bool, Dict[str, Any]]:
        """Check single text using the model.
        
        Args:
            text: Text to check
            return_score: If True, returns detailed results with score
            
        Returns:
            Boolean result or dictionary with detailed results
        """
        self._ensure_model_loaded()
        result = self.classifier([text])[0]
        is_nsfw = result['label'] == 'NSFW'
        
        if return_score:
            return {
                'flagged': is_nsfw,
                'score': result['score'] if result['label'] == 'NSFW' else 1 - result['score'],
                'label': result['label']
            }
        return is_nsfw

    def check(self, text: Union[str, List[str]], return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """Check if text contains inappropriate content using the model.
        
        Args:
            text: Single string or list of strings to check
            return_scores: If True, returns detailed results with confidence scores
            
        Returns:
            If return_scores=False:
                Boolean or list of booleans indicating NSFW content
            If return_scores=True:
                Dict or list of dicts with 'flagged', 'score', and 'label'
        """
        if isinstance(text, str):
            return self._check_single_text(text, return_scores)
            
        self._ensure_model_loaded()
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
    
    
# test
if __name__ == "__main__":
    checker = NSFW_text_classifier_Checker()
    test_texts = [
        "This is a safe text about programming",
        "Let's have a nice day at the park",
        "Fuck your mother!",
    ]
    
    # Test with simple boolean results
    results = checker.check(test_texts)
    for text, is_nsfw in zip(test_texts, results):
        print(f"Text: {text}")
        print(f"Is NSFW: {is_nsfw}\n")
    
    # Test with detailed scores
    results_with_scores = checker.check(test_texts, return_scores=True)
    for text, result in zip(test_texts, results_with_scores):
        print(f"Text: {text}")
        print(f"Result: {result}\n")