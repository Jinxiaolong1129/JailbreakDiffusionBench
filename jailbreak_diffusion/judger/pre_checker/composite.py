from typing import Union, List
from .base import BaseChecker
from .NSFW_word_match import WordMatchChecker
from .NSFW_text_classifier import ModelBasedChecker

class CompositeChecker(BaseChecker):
    """Checker that combines multiple checking methods."""
    
    def __init__(self, methods: List[str] = None):
        """Initialize with specified checking methods.
        
        Args:
            methods: List of checker names to use ['word_match', 'model']
        """
        super().__init__()
        self.methods = methods or ['word_match', 'model']
        self.checkers = {}
        
        if 'word_match' in self.methods:
            self.checkers['word_match'] = WordMatchChecker()
        if 'model' in self.methods:
            self.checkers['model'] = ModelBasedChecker()

    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check text using all specified methods.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if any method detects inappropriate content
        """
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
            
        results = []
        for t in text:
            is_inappropriate = False
            word_match_result = self.checkers['word_match'].check(t) if 'word_match' in self.methods else False
            model_result = self.checkers['model'].check(t) if 'model' in self.methods else False
            is_inappropriate = word_match_result or model_result
            results.append(is_inappropriate)
            
        return results[0] if is_single else results