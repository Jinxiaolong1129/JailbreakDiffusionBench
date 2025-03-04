# evaluation_text_detector/evaluation/__init__.py
"""
Text Detector Evaluation package for benchmarking harmful text detection models.

This package provides tools for:
- Loading and processing text datasets
- Running evaluations with different text detectors
- Calculating performance metrics
- Visualizing results
"""

from .data_loader import DatasetLoader, Prompt
from .metric import TextMetricsCalculator
from .visualization import MetricsVisualizer

__all__ = [
    'DatasetLoader',
    'Prompt',
    'TextMetricsCalculator',
    'MetricsVisualizer'
]