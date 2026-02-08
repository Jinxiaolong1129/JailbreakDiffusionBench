from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
from abc import ABC, abstractmethod
from .base import BaseChecker
import os

@dataclass
class ModerationResult:
    """Data class for storing content moderation results"""
    flagged: bool
    categories: Dict[str, bool]  # Results from the categories field
    category_scores: Dict[str, float]  # Results from the category_scores field
    category_applied_input_types: Dict[str, List[str]]  # Results from category_applied_input_types field


class OpenAITextDetector(BaseChecker):
    """Text content detector using OpenAI's moderation API."""
    
    def __init__(self, api_key: str = None, model: str = "omni-moderation-latest"):
        """Initialize the detector."""
        super().__init__()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', api_key))
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

    def detect(self, text: str) -> ModerationResult:
        """
        Detect inappropriate content in a single text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            ModerationResult: Detection results
        """
        try:
            response = self.client.moderations.create(
                model=self.model,
                input=text
            )
            
            # Get first result
            result = response.results[0]
            
            # Convert Categories object to dict using its __dict__ representation
            categories_dict = dict(result.categories)
            scores_dict = dict(result.category_scores)
            
            return ModerationResult(
                flagged=result.flagged,
                categories=categories_dict,
                category_scores=scores_dict,
                category_applied_input_types=result.category_applied_input_types
            )
            
        except Exception as e:
            raise Exception(f"Text moderation failed: {str(e)}")

    def detect_batch(self, texts: List[str]) -> List[ModerationResult]:
        """
        Detect inappropriate content in multiple texts.
        
        Args:
            texts: List of text contents to analyze
            
        Returns:
            List[ModerationResult]: List of detection results
        """
        try:
            response = self.client.moderations.create(
                model=self.model,
                input=texts
            )
            
            results = []
            for result in response.results:
                # Convert Categories object to dict for each result
                categories_dict = dict(result.categories)
                scores_dict = dict(result.category_scores)
                
                results.append(ModerationResult(
                    flagged=result.flagged,
                    categories=categories_dict,
                    category_scores=scores_dict,
                    category_applied_input_types=result.category_applied_input_types
                ))
                
            return results
            
        except Exception as e:
            raise Exception(f"Batch text moderation failed: {str(e)}")

    def _get_max_score(self, category_scores: Dict[str, float]) -> float:
        """
        Get the maximum score across all categories.
        
        Args:
            category_scores: Dictionary of category scores
            
        Returns:
            float: Maximum score
        """
        if not category_scores:
            return 0.0
        
        if category_scores.values() is None:
            return 0.0
        
        if list(category_scores.values())[0] is None:
            return 0.0
        
        # print(category_scores)
        return max(category_scores.values())

    def check(self, text: Union[str, List[str]], return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Check if content is inappropriate.
        
        Args:
            text: Single string or list of strings to check
            return_scores: If True, returns detailed results with confidence scores
            
        Returns:
            If return_scores=False:
                bool or list of bool: True if inappropriate content detected
            If return_scores=True:
                Dict or list of dicts with 'flagged', 'score', and additional data
        """
        if isinstance(text, str):
            result = self.detect(text)
            if return_scores:
                return {
                    'flagged': result.flagged,
                    'score': self._get_max_score(result.category_scores),
                    'categories': result.categories,
                    'category_scores': result.category_scores,
                    'violations': self.get_violations(result)
                }
            return result.flagged
        elif isinstance(text, list):
            results = self.detect_batch(text)
            if return_scores:
                return [
                    {
                        'flagged': result.flagged,
                        'score': self._get_max_score(result.category_scores),
                        'categories': result.categories,
                        'category_scores': result.category_scores,
                        'violations': self.get_violations(result)
                    }
                    for result in results
                ]
            return [result.flagged for result in results]
        else:
            raise TypeError("Input must be string or list of strings")
            
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
    detector = OpenAITextDetector()
    
    # Single text check
    text = "I want to kill those people."
    
    # Test with detailed scores
    result_with_scores = detector.check(text, return_scores=True)
    print(f"Text: {text}")
    print(f"Result: {result_with_scores}\n")
    
    # Multiple texts check
    texts = ["A sexy naked woman", "Oral sex beauty student", "Kill people"]
    
    # Test with simple boolean results
    results = detector.check(texts)
    for text, is_flagged in zip(texts, results):
        print(f"Text: {text}")
        print(f"Is Flagged: {is_flagged}\n")
    
    # Test with detailed scores for AUC calculation
    results_with_scores = detector.check(texts, return_scores=True)
    for text, result in zip(texts, results_with_scores):
        print(f"Text: {text}")
        print(f"Result: {result}\n")