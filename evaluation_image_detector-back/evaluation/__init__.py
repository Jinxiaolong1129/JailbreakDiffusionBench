# evaluation_image_detector/evaluation/__init__.py
from .data_loader import ImageDatasetLoader
from .metric import ImageMetricsCalculator
from .visualization import ImageMetricsVisualizer

__all__ = ['ImageDatasetLoader', 'ImageMetricsCalculator', 'ImageMetricsVisualizer']

# Version info
__version__ = '1.0.0'