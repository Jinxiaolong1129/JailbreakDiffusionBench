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
        self.data_dir = Path(data_dir)
        
    def load_dataset(self, dataset_path: str) -> List[Prompt]:
        """加载数据集"""
        with open(dataset_path) as f:
            data = json.load(f)
            
        prompts = []
        for item in data["prompts"]:
            prompt = Prompt(
                id=item["id"],
                text=item["text"],
                label=item["label"],
                source=item.get("source", ""),
                category=item.get("category", "")
            )
            prompts.append(prompt)
            
        return prompts
    
    def get_dataset_info(self, dataset_path: str) -> Dict:
        """获取数据集信息"""
        with open(dataset_path) as f:
            data = json.load(f)
        return {
            "total_prompts": len(data["prompts"]),
            "malicious_prompts": sum(1 for p in data["prompts"] if p["label"] == "malicious"),
            "benign_prompts": sum(1 for p in data["prompts"] if p["label"] == "benign"),
            "categories": list(set(p.get("category", "") for p in data["prompts"]))
        }
