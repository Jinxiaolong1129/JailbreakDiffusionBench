from dataclasses import dataclass
from typing import Dict, List, Optional, Union
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

    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Check if content is inappropriate.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        if isinstance(text, str):
            result = self.detect(text)
            return result.flagged
        elif isinstance(text, list):
            results = self.detect_batch(text) # TODO save all results here
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
                if result.categories[category]]

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
    text = "Fuck your mother."
    result = detector.detect(text)
    print("Moderation results:", result)
    
    if detector.is_safe(result):
        print("Content is safe")
    else:
        print("Violations detected:", detector.get_violations(result))
    
    # Multiple texts check
    texts = ["A sexy naked woman", "Oral sex beauty student", "Kill people"]
    results = detector.check(texts)
    print("Batch check results:", results)