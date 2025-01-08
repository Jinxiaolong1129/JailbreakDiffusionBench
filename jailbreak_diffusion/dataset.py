from typing import Literal, Optional, List, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset

class DiffusionJailbreakDataset(Dataset):
    """Dataset for jailbreaking diffusion models.
    
    Collects and organizes data that could potentially lead to model jailbreak,
    including text prompts and corresponding targets.
    
    The dataset contains:
        - prompts/images: Input prompts or images
        - targets: Expected output responses
        - behaviors: Behavior type identifiers
        - categories: Behavior categories (e.g., "Physical harm")
        - sources: Data sources
    """
    # TODO: Update this with your Hugging Face account name
    HF_ACCOUNT = "dedeswim"  
    
    def __init__(
        self,
        split: Literal["harmful", "benign"],
        include_images: bool = False,
        image_size: Optional[int] = None
    ):
        """Initialize the dataset.
        
        Args:
            split: Choose between "harmful" or "benign" dataset
            include_images: Whether to include image data
            image_size: If including images, specify the image size
        """
        # Load base dataset
        dataset = load_dataset(
            f"{self.HF_ACCOUNT}/JBB-Behaviors",
            "behaviors",
            split=split
        )
        self.df = dataset.to_pandas().drop(columns=["Index"])
        
        # Store text data
        self.prompts = self.df.Goal.to_list()  # jailbreak prompts
        self.targets = self.df.Target.to_list()  # target responses
        self.behaviors = self.df.Behavior.to_list()  # behavior identifiers
        self.categories = self.df.Category.to_list()  # categories
        self.sources = self.df.Source.to_list()  # sources
        
        # Image-related settings (if needed)
        self.include_images = include_images
        self.image_size = image_size
        if include_images:
            assert image_size is not None, "Must specify image_size when including images"
            self.images = self._prepare_images()
    
    def _prepare_images(self) -> List[torch.Tensor]:
        """Prepare or generate image data corresponding to prompts"""
        # Implement image generation or loading logic here
        # Currently returns empty list as placeholder
        return []
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single data sample
        
        Returns:
            Dictionary containing:
            - prompt: Input prompt
            - target: Target response
            - behavior: Behavior type
            - category: Category
            - source: Source
            - image: If include_images=True
        """
        sample = {
            'prompt': self.prompts[idx],
            'target': self.targets[idx],
            'behavior': self.behaviors[idx],
            'category': self.categories[idx],
            'source': self.sources[idx]
        }
        
        if self.include_images and self.images:
            sample['image'] = self.images[idx]
            
        return sample

    def get_behavior_stats(self) -> Dict[str, int]:
        """Get statistics of different behavior types in the dataset"""
        from collections import Counter
        return dict(Counter(self.behaviors))
    
    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics of different categories in the dataset"""
        from collections import Counter
        return dict(Counter(self.categories))



def load_jailbreak_dataset(
    split: Literal["harmful", "benign"] = "harmful",
    include_images: bool = False,
    image_size: Optional[int] = None
) -> DiffusionJailbreakDataset:
    """Convenience function to load the jailbreak dataset"""
    return DiffusionJailbreakDataset(split, include_images, image_size)