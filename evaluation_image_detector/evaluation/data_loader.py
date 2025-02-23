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
        """加载图像数据集"""
        with open(dataset_path) as f:
            data = json.load(f)
            
        items = []
        for item in data["images"]:
            image_path = self.data_dir / item["file_path"]
            image = Image.open(image_path)
            image_item = ImageItem(
                id=item["id"],
                image=image,
                label=item["label"],
                source=item.get("source", ""),
                category=item.get("category", "")
            )
            items.append(image_item)
            
        return items
    
    def get_dataset_info(self, dataset_path: str) -> Dict:
        """获取数据集信息"""
        with open(dataset_path) as f:
            data = json.load(f)
        return {
            "total_images": len(data["images"]),
            "unsafe_images": sum(1 for img in data["images"] if img["label"] == "unsafe"),
            "safe_images": sum(1 for img in data["images"] if img["label"] == "safe"),
            "categories": list(set(img.get("category", "") for img in data["images"]))
        }