from typing import Union, List, Dict, Any
import numpy as np
from detoxify import Detoxify

from base_checker import BaseChecker  # Assuming base_checker.py contains the BaseChecker class


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
    
    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if text contains inappropriate content based on toxicity scores.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if any toxicity score exceeds the threshold
        """
        # Get prediction scores from Detoxify
        results = self.model.predict(text)
        
        # Handle both single text and list of texts
        if isinstance(text, str):
            # For a single text, check if any category exceeds threshold
            return self._check_single_result(results)
        else:
            # For multiple texts, check each one
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


# Example usage
if __name__ == "__main__":
    # Create a checker with a threshold of 0.7
    checker = DetoxifyChecker(threshold=0.7)
    
    # Check a single text
    text = "This is a neutral sentence."
    result = checker(text)
    print(f"Text: '{text}'")
    print(f"Contains inappropriate content: {result}")
    
    # Check multiple texts
    texts = ["This is fine.", "Fuck your mother!"]
    results = checker(texts)
    for t, r in zip(texts, results):
        print(f"Text: '{t}'")
        print(f"Contains inappropriate content: {r}")