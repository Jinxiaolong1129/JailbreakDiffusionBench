# jailbreak_diffusion/judger/pre_checker/distilbert-nsfw-text-classifier.py
# eliasalbouzidi/distilbert-nsfw-text-classifier


from typing import Union, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .base import BaseChecker

class distilbert_nsfw_text_checker(BaseChecker):
    """Content checker using DistilBERT model."""
    
    def __init__(self):
        """Initialize the checker."""
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("eliasalbouzidi/distilbert-nsfw-text-classifier")
        self.model = AutoModelForSequenceClassification.from_pretrained("eliasalbouzidi/distilbert-nsfw-text-classifier")
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        
    def _check_single_text(self, text: str) -> bool:
        """Check single text using DistilBERT."""
        try:
            result = self.classifier(text)[0]
            return result['label'] == 'NSFW'
        except Exception as e:
            print(f"DistilBERT error: {str(e)}")
            return False
            
    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check text(s) for inappropriate content."""
        if isinstance(text, str):
            return self._check_single_text(text)
        return [self._check_single_text(t) for t in text]
    
    
# Usage example
if __name__ == "__main__":
    # Initialize checker
    checker = distilbert_nsfw_text_checker()
    
    # Check single text
    result = checker.check("Fuck your mother.")
    print("Inappropriate content detected:", result)
    
    # Check multiple texts
    results = checker.check(["This is a test message.", "This is another message."])
    print("Results:", results)