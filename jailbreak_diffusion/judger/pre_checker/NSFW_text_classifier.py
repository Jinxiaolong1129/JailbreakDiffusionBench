# jailbreak_diffusion/judger/pre_checker/NSFW_text_classifier.py

from typing import Union, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .base import BaseChecker


# NOTE finish testing
# True: NSFW
# False: Safe

class ModelBasedChecker(BaseChecker):
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

    def _check_single_text(self, text: str) -> bool:
        """Check single text using the model."""
        self._ensure_model_loaded()
        result = self.classifier([text])[0]
        return result['label'] == 'NSFW'

    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if text contains inappropriate content using the model.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        if isinstance(text, str):
            return self._check_single_text(text)
            
        self._ensure_model_loaded()
        results = self.classifier(text)
        return [result['label'] == 'NSFW' for result in results]
    
    
# test
if __name__ == "__main__":
    checker = ModelBasedChecker()
    test_texts = [
        "This is a safe text about programming",
        "Let's have a nice day at the park",
        "Fuck your mother!",
    ]
    results = checker.check(test_texts)
    for text, is_nsfw in zip(test_texts, results):
        print(f"Text: {text}")
        print(f"Is NSFW: {is_nsfw}\n")