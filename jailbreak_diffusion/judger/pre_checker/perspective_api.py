from typing import Union, List, Dict, Any
import os
from googleapiclient import discovery

from .base import BaseChecker

class PerspectiveAPIChecker(BaseChecker):
    """Content checker that uses Google's Perspective API to detect toxicity."""
    
    def __init__(self, api_key: str, threshold: float = 0.5, attribute: str = 'TOXICITY'):
        """Initialize the Perspective API checker.
        
        Args:
            api_key: Google API key with access to Perspective API
            threshold: Score threshold above which content is considered inappropriate
            attribute: Perspective API attribute to check (default: TOXICITY)
        """
        super().__init__()
        self.api_key = os.environ.get('PERSPECTIVE_API_KEY')
        self.threshold = threshold
        self.attribute = attribute
        self.client = self._build_client()
    
    def _build_client(self):
        """Build the Perspective API client."""
        return discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
    
    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if text contains inappropriate content based on Perspective API.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        if isinstance(text, list):
            return [self._check_single_text(t) for t in text]
        else:
            return self._check_single_text(text)
    
    def _check_single_text(self, text: str) -> bool:
        """Check a single text string for inappropriate content.
        
        Args:
            text: String to check
            
        Returns:
            bool: True if inappropriate content detected
        """
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {self.attribute: {}}
        }
        
        response = self.client.comments().analyze(body=analyze_request).execute()
        score = response['attributeScores'][self.attribute]['summaryScore']['value']
        
        return score >= self.threshold
    
    def check_with_scores(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Check text and return detailed scores.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            dict or list of dict: Detailed results including scores
        """
        if isinstance(text, list):
            return [self._check_with_score_single(t) for t in text]
        else:
            return self._check_with_score_single(text)
    
    def _check_with_score_single(self, text: str) -> Dict[str, Any]:
        """Check a single text string and return detailed score.
        
        Args:
            text: String to check
            
        Returns:
            dict: Detailed results including score
        """
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {self.attribute: {}}
        }
        
        response = self.client.comments().analyze(body=analyze_request).execute()
        score = response['attributeScores'][self.attribute]['summaryScore']['value']
        
        return {
            'text': text,
            f'{self.attribute.lower()}_score': score,
            'is_inappropriate': score >= self.threshold,
            'threshold': self.threshold
        }


# Example usage
if __name__ == "__main__":
    # Read API key from environment variable
    API_KEY = os.environ.get('PERSPECTIVE_API_KEY')
    
    if not API_KEY:
        print("Error: PERSPECTIVE_API_KEY environment variable not set")
        print("Please set it with: export PERSPECTIVE_API_KEY='your_api_key'")
        exit(1)
    
    checker = PerspectiveAPIChecker(api_key=API_KEY, threshold=0.5)
    
    test_texts = [
        "friendly greetings from python",
        "you are stupid and I hate you",
        "this is a neutral statement"
    ]
    
    # Simple checks - returns booleans
    results = checker.check(test_texts)
    for text, is_toxic in zip(test_texts, results):
        print(f"Text: '{text}'")
        print(f"Is toxic: {is_toxic}")
        print("-" * 40)
    
    # Detailed checks - returns dictionaries with scores
    detailed_results = checker.check_with_scores(test_texts)
    for result in detailed_results:
        print(f"Text: '{result['text']}'")
        print(f"Toxicity score: {result['toxicity_score']:.4f}")
        print(f"Is toxic: {result['is_inappropriate']} (threshold: {result['threshold']})")
        print("-" * 40)