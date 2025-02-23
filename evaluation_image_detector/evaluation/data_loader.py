from typing import List, Dict
from pathlib import Path
import json
from dataclasses import dataclass
from PIL import Image

@dataclass
class ImageItem:
    id: str
    image: Image.Image
    label: str
    source: str
    category: str = ""
    
class ImageDatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_dataset(self, dataset_path: str) -> List[ImageItem]:
        """Load dataset of images and their metadata"""
        dataset_dir = Path(dataset_path).parent
        with open(dataset_path) as f:
            data = json.load(f)
            
        items = []
        for item in data["images"]:
            # Load image
            image_path = dataset_dir / item["file_name"]
            try:
                image = Image.open(image_path).convert('RGB')
                
                item_obj = ImageItem(
                    id=item["id"],
                    image=image,
                    label=item["label"],
                    source=item.get("source", ""),
                    category=item.get("category", "")
                )
                items.append(item_obj)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                
        return items
    
    def get_dataset_info(self, dataset_path: str) -> Dict:
        """Get dataset information"""
        with open(dataset_path) as f:
            data = json.load(f)
        return {
            "total_images": len(data["images"]),
            "unsafe_images": sum(1 for img in data["images"] if img["label"] == "unsafe"),
            "safe_images": sum(1 for img in data["images"] if img["label"] == "safe"),
            "categories": list(set(img.get("category", "") for img in data["images"])),
            "image_formats": list(set(Path(img["file_name"]).suffix for img in data["images"]))
        }
        
    def get_category_distribution(self, dataset_path: str) -> Dict[str, Dict[str, int]]:
        """Get distribution of safe/unsafe images across categories"""
        with open(dataset_path) as f:
            data = json.load(f)
            
        distribution = {}
        for img in data["images"]:
            category = img.get("category", "uncategorized")
            if category not in distribution:
                distribution[category] = {"safe": 0, "unsafe": 0}
            distribution[category][img["label"]] += 1
            
        return distribution