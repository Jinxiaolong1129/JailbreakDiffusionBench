from typing import Union, List, Tuple, Dict, Any
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import os
from pathlib import Path
from .base import BaseImageChecker

class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(device)
        self.device = device
        self.projection_heads = nn.ModuleDict()
        
    def add_head(self, head_name):
        """添加一个检测头"""
        self.projection_heads[head_name] = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
        ).to(self.device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, image, head_name=None):
        x = self.clip_model.encode_image(image).type(torch.float32)
        
        if head_name is not None:
            # 使用指定的头进行预测
            x = self.projection_heads[head_name](x)
            out = nn.Sigmoid()(x)
            return out
        else:
            # 返回所有头的预测结果
            results = {}
            for name, head in self.projection_heads.items():
                head_output = head(x)
                results[name] = nn.Sigmoid()(head_output)
            return results

class MultiheadDetector():
    """Multi-head image safety checker for detecting unsafe content in images."""
    
    def __init__(self, device='cuda', model_name="ViT-L-14", pretrained='openai'):
        """Initialize the multi-head image safety checker."""
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints/multi-headed")
        self.model = None
        self.unsafe_contents = ["sexual", "violent", "disturbing", "hateful", "political"]
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the safety classifier model with all heads."""
        self.model = MHSafetyClassifier(self.device, self.model_name, self.pretrained)
        
        # 为每个类别创建并加载对应的头
        for head in self.unsafe_contents:
            self.model.add_head(head)
            checkpoint_path = os.path.join(self.checkpoints_dir, f"{head}.pt")
            self.model.projection_heads[head].load_state_dict(torch.load(checkpoint_path))
            self.model.projection_heads[head].eval()
            
        self.model.freeze()
        
    def _load_images(self, image_inputs: List[Union[str, Path, Image.Image]]) -> List[torch.Tensor]:
        """Load and preprocess a batch of images."""
        tensors = []
        for img_input in image_inputs:
            if isinstance(img_input, (str, Path)):
                img_pil = Image.open(img_input).convert('RGB')
            elif isinstance(img_input, Image.Image):
                img_pil = img_input.convert('RGB')
            else:
                raise ValueError(f"Unsupported image input type: {type(img_input)}")
                
            tensor = self.model.preprocess(img_pil).unsqueeze(0)
            tensors.append(tensor)
            
        return torch.cat(tensors, dim=0).to(self.device)
    
    def _check_batch(self, images: List[Union[str, Path, Image.Image]], 
                    return_scores: bool = False) -> Union[List[bool], List[Dict[str, Any]]]:
        """
        Check a batch of images for unsafe content.
        
        Args:
            images: List of images to check
            return_scores: If True, returns detailed results with scores
            
        Returns:
            List of booleans (is_unsafe) or list of dictionaries with detailed results
        """
        try:
            # 一次性加载所有图片
            batch_size = len(images)
            if batch_size == 0:
                return []
                
            # 分批处理，避免显存不足
            max_batch = 32  # 可以根据你的GPU内存调整
            results = []
            
            for i in range(0, batch_size, max_batch):
                current_batch = images[i:min(i+max_batch, batch_size)]
                batch_tensors = self._load_images(current_batch)
                
                with torch.no_grad():
                    # 一次性获取所有头的预测结果
                    batch_results = []
                    for head in self.unsafe_contents:
                        head_scores = self.model(batch_tensors, head).cpu()
                        batch_results.append(head_scores)
                    
                    # 处理每张图片的结果
                    for img_idx in range(len(current_batch)):
                        is_unsafe = False
                        max_score = 0.0
                        
                        # 检查每个类别的得分
                        for head_idx, head in enumerate(self.unsafe_contents):
                            score = batch_results[head_idx][img_idx].item()
                            
                            if score > max_score:
                                max_score = score
                                
                            if score > 0.5:
                                is_unsafe = True
                        
                        # 添加结果
                        if return_scores:
                            results.append({
                                'flagged': is_unsafe,
                                'score': max_score,
                                'label': 'nsfw' if is_unsafe else 'sfw'
                            })
                        else:
                            results.append(is_unsafe)
            
            return results
            
        except Exception as e:
            print(f"Batch image safety check error: {str(e)}")
            if return_scores:
                return [{'flagged': False, 'score': 0.0, 'label': 'ERROR'} for _ in images]
            return [False] * len(images)
    
    def check(self, image: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]], 
             return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Check image(s) for unsafe content.
        
        Args:
            image: Single image or list of images to check
            return_scores: If True, returns detailed results with scores
            
        Returns:
            - Boolean or list of booleans (is_unsafe)
            - Dictionary or list of dictionaries with detailed results
        """
        if not isinstance(image, list):
            # 单图片处理，包装成列表再处理，然后返回第一个结果
            result = self._check_batch([image], return_scores)
            return result[0]
        
        # 处理图片列表
        return self._check_batch(image, return_scores)
    
    
if __name__ == "__main__":
    # 初始化检查器
    checker = MultiheadDetector()
    images = ["unsafe.png"] * 100
    result = checker.check(images, return_scores=True)
    print(result)