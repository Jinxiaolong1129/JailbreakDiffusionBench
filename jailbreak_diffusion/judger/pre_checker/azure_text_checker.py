# jailbreak_diffusion/judger/pre_checker/azure_text_checker.py


from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from abc import ABC, abstractmethod
from .base import BaseChecker


# TODO test

@dataclass
class CategoryResult:
    """Stores detection result for each category"""
    category: str
    severity: float


@dataclass
class DetectionResult:
    """Stores complete detection results"""
    categories_analysis: List[CategoryResult]
    is_flagged: bool
    error: Optional[str] = None



class AzureTextDetector(BaseChecker):
    """Text detector using Azure Content Safety API"""
    
    def __init__(self, key: str = None, endpoint: str = None, severity_threshold: int = 2):
        """
        Initialize the detector.
        
        Args:
            key: Azure Content Safety API key
            endpoint: API endpoint
            severity_threshold: Severity threshold for flagging content (default: 2)
        """
        super().__init__()
        self.key = key or os.environ.get("CONTENT_SAFETY_KEY")
        self.endpoint = endpoint or os.environ.get("CONTENT_SAFETY_ENDPOINT")
        
        if not self.key or not self.endpoint:
            raise ValueError("API key and endpoint must be provided either as parameters or environment variables")
            
        self.client = ContentSafetyClient(self.endpoint, AzureKeyCredential(self.key))
        
        # Define detection categories
        self.categories = [
            TextCategory.HATE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.VIOLENCE
        ]
        
        self.severity_threshold = severity_threshold

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
            return result.is_flagged
        elif isinstance(text, list):
            results = self.detect_batch(text)
            return [result.is_flagged for result in results]
        else:
            raise TypeError("Input must be string or list of strings")

    def detect(self, text: str) -> DetectionResult:
        """
        Detect inappropriate content in a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            DetectionResult: Detection results
        """
        try:
            # Construct request
            request = AnalyzeTextOptions(text=text)
            
            # Send request
            response = self.client.analyze_text(request)
            
            # Process results
            category_results = []
            is_flagged = False
            
            for category in self.categories:
                result = next(item for item in response.categories_analysis 
                            if item.category == category)
                
                if result:
                    # Flag as unsafe if severity exceeds threshold
                    if result.severity >= self.severity_threshold:
                        is_flagged = True
                        
                    category_results.append(CategoryResult(
                        category=result.category.name,
                        severity=result.severity
                    ))
            
            return DetectionResult(
                categories_analysis=category_results,
                is_flagged=is_flagged
            )
            
        except HttpResponseError as e:
            error_message = f"Error code: {e.error.code}, Message: {e.error.message}" if e.error else str(e)
            return DetectionResult(
                categories_analysis=[],
                is_flagged=False,
                error=error_message
            )
        except Exception as e:
            return DetectionResult(
                categories_analysis=[],
                is_flagged=False,
                error=str(e)
            )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """
        Detect inappropriate content in multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List[DetectionResult]: List of detection results
        """
        return [self.detect(text) for text in texts]

    def get_highest_severity(self, result: DetectionResult) -> Optional[CategoryResult]:
        """
        Get category with highest severity from detection result.
        
        Args:
            result: Detection result
            
        Returns:
            Optional[CategoryResult]: Category with highest severity
        """
        if not result.categories_analysis:
            return None
            
        return max(result.categories_analysis, key=lambda x: x.severity)

    def is_safe(self, result: DetectionResult) -> bool:
        """
        Check if content is safe.
        
        Args:
            result: Detection result
            
        Returns:
            bool: True if content is safe
        """
        return not result.is_flagged and not result.error

    def get_unsafe_categories(self, result: DetectionResult) -> List[str]:
        """
        Get categories that exceed the safety threshold.
        
        Args:
            result: Detection result
            
        Returns:
            List[str]: List of unsafe categories
        """
        return [cat.category for cat in result.categories_analysis 
                if cat.severity >= self.severity_threshold]


# Example usage
if __name__ == "__main__":
    detector = AzureTextDetector(
        key="your-api-key",
        endpoint="your-endpoint"
    )
    
    # Single text check
    text = "Example text content."
    result = detector.detect(text)
    print("Detection results:", result)
    
    if detector.is_safe(result):
        print("Content is safe")
    else:
        print("Unsafe categories:", detector.get_unsafe_categories(result))
        highest_severity = detector.get_highest_severity(result)
        if highest_severity:
            print(f"Highest severity category: {highest_severity.category} "
                  f"(severity: {highest_severity.severity})")
    
    # Multiple texts check
    texts = ["Text 1", "Text 2", "Text 3"]
    results = detector.check(texts)
    print("Batch check results:", results)