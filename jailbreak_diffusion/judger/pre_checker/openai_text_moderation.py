# jailbreak_diffusion/judger/pre_checker/openai_text_moderation.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from openai import OpenAI
from abc import ABC, abstractmethod
from .base import BaseChecker
import os

# NOTE finish testing

# True: NSFW
# False: Safe

@dataclass
class ModerationResult:
    """Data class for storing content moderation results"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Optional[Dict[str, List[str]]] = None


class OpenAITextDetector(BaseChecker):
    """Text content detector using OpenAI's moderation API."""
    
    def __init__(self, api_key: str = None, model: str = "text-moderation-latest"):
        """
        Initialize the detector.
        
        Args:
            api_key: OpenAI API key
            model: Model to use, defaults to latest text moderation model
        """
        super().__init__()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', api_key))
        self.model = model
        
        # All possible moderation categories
        self.categories = [
            "harassment", "harassment/threatening",
            "hate", "hate/threatening",
            "illicit", "illicit/violent",
            "self-harm", "self-harm/intent", "self-harm/instructions",
            "sexual", "sexual/minors",
            "violence", "violence/graphic"
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
            
            # Get first result (since only one text was input)
            result = response.results[0]
            
            return ModerationResult(
                flagged=result.flagged,
                categories=result.categories,
                category_scores=result.category_scores,
                category_applied_input_types=getattr(result, 'category_applied_input_types', None)
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
                # TODO: 更多的检测结果
                results.append(ModerationResult(
                    flagged=result.flagged,
                    categories=result.categories,
                    category_scores=result.category_scores,
                    category_applied_input_types=getattr(result, 'category_applied_input_types', None)
                ))
                
            return results
            
        except Exception as e:
            raise Exception(f"Batch text moderation failed: {str(e)}")

    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Implementation of BaseChecker's check method.
        Returns True if inappropriate content is detected.

        Args:
            text: Single string or list of strings to check

        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        if isinstance(text, str):
            result = self.detect(text)
            return result.flagged
        elif isinstance(text, list):
            results = self.detect_batch(text)
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
                if result.categories.get(category, False)]

    def is_safe(self, result: ModerationResult) -> bool:
        """
        Check if content is safe (no violations detected).
        
        Args:
            result: Detection result
            
        Returns:
            bool: True if content is safe
        """
        return not result.flagged


# Example usage
if __name__ == "__main__":
    detector = OpenAITextDetector()
    
    # Single text check
    text = "Example text content."
    result = detector.detect(text)
    print("Moderation results:", result)
    
    if detector.is_safe(result):
        print("Content is safe")
    else:
        print("Violations detected:", detector.get_violations(result))
    
    # Multiple texts check
    texts = ["Text 1", "Text 2", "Text 3"]
    results = detector.check(texts)
    print("Batch check results:", results)