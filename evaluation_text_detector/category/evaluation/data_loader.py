# evaluation_text_detector/evaluation/data_loader.py
from typing import List, Dict
from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class Prompt:
    id: str
    text: str
    label: str
    source: str
    category: str = ""

class DatasetLoader:
    def __init__(self, data_dir: str):
        """Initialize dataset loader with base data directory"""
        self.data_dir = Path(data_dir)
        
    def load_dataset(self, dataset_path: str) -> List[Prompt]:
        """Load dataset from specified path"""
        # Ensure dataset_path is an absolute path or resolve against data_dir
        path = Path(dataset_path)
            
        with open(path) as f:
            data = json.load(f)
            
        prompts = []
        for item in data["prompts"]:
            prompt = Prompt(
                id=item["id"],
                text=item["text"],
                label=item["label"],  # assuming 'harmful' or 'benign'
                source=item.get("source", ""),
                category=item.get("category", "")
            )
            prompts.append(prompt)
            
        return prompts
    
    def get_dataset_info(self, dataset_path: str) -> Dict:
        """Get dataset statistics and information"""
        # Ensure dataset_path is an absolute path or resolve against data_dir
        path = Path(dataset_path)
            
        with open(path) as f:
            data = json.load(f)
            
        # Count different types of prompts
        harmful_count = sum(1 for p in data["prompts"] if p["label"] == "harmful")
        benign_count = sum(1 for p in data["prompts"] if p["label"] == "benign")
        
        # Get all unique categories
        categories = {}
        for p in data["prompts"]:
            category = p.get("category", "")
            if category:
                # Handle case where category is a list
                if isinstance(category, list):
                    for cat in category:
                        categories[cat] = categories.get(cat, 0) + 1
                else:
                    categories[category] = categories.get(category, 0) + 1
        
        # Get sources
        sources = {}
        for p in data["prompts"]:
            source = p.get("source", "")
            if source:
                sources[source] = sources.get(source, 0) + 1
                
        return {
            "name": data.get("name", Path(dataset_path).stem),
            "description": data.get("description", ""),
            "total_prompts": len(data["prompts"]),
            "harmful_prompts": harmful_count,
            "benign_prompts": benign_count,
            "categories": {
                "unique": list(categories.keys()),
                "counts": categories
            },
            "sources": {
                "unique": list(sources.keys()),
                "counts": sources
            }
        }