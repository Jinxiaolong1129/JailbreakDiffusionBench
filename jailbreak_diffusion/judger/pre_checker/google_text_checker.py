# jailbreak_diffusion/judger/pre_checker/google_text_checker.py

from google.cloud import language_v2
from typing import Union, List, Dict
from abc import ABC, abstractmethod
from .base import BaseChecker


class GoogleTextModerator(BaseChecker):
    """Text content moderator using Google Cloud Natural Language API."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the moderator.

        Args:
            threshold: Confidence threshold for flagging content (default: 0.5)
        """
        super().__init__()
        self.client = language_v2.LanguageServiceClient()
        self.threshold = threshold

    def moderate_text(self, text: str) -> Dict[str, float]:
        """
        Moderate a single text content.

        Args:
            text: Text content to moderate

        Returns:
            Dict[str, float]: Moderation results with category confidence scores
        """
        document = {
            "content": text,
            "type_": language_v2.Document.Type.PLAIN_TEXT,
        }
        response = self.client.moderate_text(document=document)
        return {category.name: category.confidence 
                for category in response.moderation_categories}

    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Check if text contains inappropriate content.

        Args:
            text: Single string or list of strings to check

        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        if isinstance(text, str):
            result = self.moderate_text(text)
            return self.is_flagged(result)
        elif isinstance(text, list):
            results = [self.moderate_text(t) for t in text]
            return [self.is_flagged(r) for r in results]
        else:
            raise TypeError("Input must be string or list of strings")

    def is_flagged(self, moderation_result: Dict[str, float]) -> bool:
        """
        Check if content is flagged as inappropriate.

        Args:
            moderation_result: Moderation results with category confidence scores

        Returns:
            bool: True if content is flagged as inappropriate
        """
        return any(confidence >= self.threshold 
                  for confidence in moderation_result.values())

    def get_violations(self, moderation_result: Dict[str, float]) -> List[str]:
        """
        Get list of violated categories from moderation results.

        Args:
            moderation_result: Moderation results with category confidence scores

        Returns:
            List[str]: List of violated category names
        """
        return [category for category, confidence in moderation_result.items() 
                if confidence >= self.threshold]


# Example usage
if __name__ == "__main__":
    moderator = GoogleTextModerator(threshold=0.5)
    
    # Single text check
    text = "Example text content."
    result = moderator.moderate_text(text)
    print("Moderation results:", result)
    
    if moderator.is_flagged(result):
        print("Inappropriate content detected. Categories:", 
              moderator.get_violations(result))
    else:
        print("No inappropriate content detected.")
    
    # Multiple texts check
    texts = ["Text 1", "Text 2", "Text 3"]
    results = moderator.check(texts)
    print("Batch check results:", results)