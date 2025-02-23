from abc import ABC, abstractmethod
from typing import Union, List

class BaseChecker(ABC):
    """Base class for all text content checkers."""
    
    def __init__(self):
        """Initialize the checker."""
        pass

    @abstractmethod
    def check(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if text contains inappropriate content.
        
        Args:
            text: Single string or list of strings to check
            
        Returns:
            bool or list of bool: True if inappropriate content detected
        """
        pass

    def __call__(self, text: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Convenience method to check text."""
        return self.check(text)