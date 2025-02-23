from typing import Union, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from .base import BaseChecker

class ModelBasedChecker(BaseChecker):
    """Checker that uses a pre-trained model to detect inappropriate content."""
    # NSFW 返回 True 
    
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
    
    
    