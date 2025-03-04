from typing import Union, List, Dict, Any
import numpy as np
from detoxify import Detoxify

from .base import BaseChecker

class DetoxifyChecker(BaseChecker):
    """Toxicity checker using the Detoxify model."""
    
    def __init__(self, threshold: float = 0.5, model_type: str = 'original', device: str = 'cpu'):
        """Initialize the Detoxify checker.
        
        Args:
            threshold: Threshold value to determine if content is inappropriate
            model_type: Type of Detoxify model to use ('original', 'unbiased', etc.)
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        self.threshold = threshold
        self.model = Detoxify(model_type, device=device)
    
    def check(self, text: Union[str, List[str]], return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """Check if text contains inappropriate content based on toxicity scores.
        
        Args:
            text: Single string or list of strings to check
            return_scores: If True, returns detailed results with scores
            
        Returns:
            If return_scores=False:
                bool or list of bool: True if any toxicity score exceeds the threshold
            If return_scores=True:
                Dict or list of dicts with detailed toxicity scores and flagging result
        """
        # Get prediction scores from Detoxify
        results = self.model.predict(text)
        
        # Handle both single text and list of texts
        if isinstance(text, str):
            # For a single text
            if return_scores:
                return self._format_single_result(results, self._check_single_result(results))
            else:
                return self._check_single_result(results)
        else:
            # For multiple texts
            if return_scores:
                return [self._format_single_result(self._get_single_result(results, i), 
                       self._check_single_result(self._get_single_result(results, i)))
                       for i in range(len(text))]
            else:
                return [self._check_single_result(self._get_single_result(results, i))
                        for i in range(len(text))]
    
    def _check_single_result(self, result: Dict[str, Any]) -> bool:
        """Check if any toxicity category in a single result exceeds the threshold.
        
        Args:
            result: Dictionary of toxicity scores for a single text
            
        Returns:
            bool: True if any toxicity score exceeds the threshold
        """
        # Check if any category's score is above the threshold
        return any(score > self.threshold for score in result.values())
    
    def _get_single_result(self, results: Dict[str, List[float]], index: int) -> Dict[str, float]:
        """Extract results for a single text from batch results.
        
        Args:
            results: Dictionary of toxicity scores for multiple texts
            index: Index of the text to extract scores for
            
        Returns:
            Dict[str, float]: Dictionary of toxicity scores for a single text
        """
        return {key: values[index] for key, values in results.items()}
    
    def _format_single_result(self, result: Dict[str, float], is_flagged: bool) -> Dict[str, Any]:
        """Format a single result for detailed output.
        
        Args:
            result: Dictionary of toxicity scores for a single text
            is_flagged: Whether the text was flagged as inappropriate
            
        Returns:
            Dict with 'flagged', 'score', and detailed toxicity scores
        """
        # Get the maximum toxicity score for the primary score
        max_score = max(result.values()) if result else 0.0
        
        return {
            'flagged': is_flagged,
            'score': max_score,  # Primary score for AUC calculation
            'toxicity_scores': result,
            'threshold': self.threshold
        }


# Example usage
if __name__ == "__main__":
    # Create a checker with a threshold of 0.7
    checker = DetoxifyChecker(threshold=0.7)
    
    # Check a single text
    text = "This is a neutral sentence."
    result = checker.check(text)
    print(f"Text: '{text}'")
    print(f"Contains inappropriate content: {result}")
    
    # Check with detailed scores
    detailed = checker.check(text, return_scores=True)
    print(f"Detailed results: {detailed}")
    
    # Check multiple texts
    texts = ["This is fine.", "Fuck your mother!"]
    results = checker.check(texts)
    for t, r in zip(texts, results):
        print(f"Text: '{t}'")
        print(f"Contains inappropriate content: {r}")
    
    # Check multiple texts with detailed scores
    detailed_results = checker.check(texts, return_scores=True)
    for t, r in zip(texts, detailed_results):
        print(f"Text: '{t}'")
        print(f"Detailed results: {r}")