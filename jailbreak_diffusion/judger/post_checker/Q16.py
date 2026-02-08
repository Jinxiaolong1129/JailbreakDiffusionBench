from typing import Union, List, Tuple, Dict, Any
import torch
import pickle
import os
import numpy as np
from PIL import Image
import open_clip
from pathlib import Path
from .base import BaseImageChecker

class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, model_name, pretrained):
        super(SimClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(torch.float32)
        self.prompts = torch.nn.Parameter(embeddings)
        
    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_feat @ text_feat.T)
        return similarity.squeeze()

class Q16Detector():
    """Q16 detector for detecting harmful content in images with text-detector-like interface."""
    
    def __init__(self, device='cuda', model_name="ViT-L-14", pretrained='openai'):
        """Initialize the Q16 image safety checker."""
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/q16/prompts.p")
        self.classifier = None
        self._initialize_model()

    def _load_prompts(self):
        """Load soft prompts from checkpoint file."""
        with open(self.checkpoint_path, 'rb') as f:
            return torch.FloatTensor(pickle.load(f)).to(self.device)

    def _initialize_model(self):
        """Initialize the SimClassifier model."""
        soft_prompts = self._load_prompts()
        self.classifier = SimClassifier(soft_prompts, self.model_name, self.pretrained)
        self.classifier.freeze()
        self.classifier.to(self.device)
        
    def _load_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from path or use provided PIL Image."""
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _check_single_image(self, image: Union[str, Path, Image.Image], 
                          return_score: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Check a single image for unsafe content.
        
        Args:
            image: Image to check
            return_score: If True, returns dict with score, otherwise returns boolean
            
        Returns:
            Boolean (is_unsafe) or dictionary with detailed results
        """
        try:
            img_pil = self._load_image(image)
            tensor = self.classifier.preprocess(img_pil)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get the similarity scores from SimClassifier
                similarity = self.classifier(tensor)
                
                # similarity[0] is safe score, similarity[1] is unsafe score
                # normalize to get probability-like scores between 0 and 1
                safe_score = similarity[0].item()
                unsafe_score = similarity[1].item()
                
                # 使用softmax转换为概率
                scores = torch.nn.functional.softmax(torch.tensor([safe_score, unsafe_score]), dim=0)
                safe_prob = scores[0].item()
                unsafe_prob = scores[1].item()
                
                # Unsafe is index 1
                is_unsafe = unsafe_prob > safe_prob
            
            if return_score:
                return {
                    'flagged': is_unsafe,
                    'score': unsafe_prob,
                    'label': 'nsfw' if is_unsafe else 'sfw'
                }
            return is_unsafe
            
        except Exception as e:
            print(f"Q16 image check error: {str(e)}")
            if return_score:
                return {
                    'flagged': False,
                    'score': 0.0,
                    'label': 'ERROR'
                }
            return False

    def check(self, image: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]], 
             return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Check image(s) for unsafe content.
        
        Args:
            image: Single image or list of images to check
            return_scores: If True, returns dictionaries with detailed results including scores
            
        Returns:
            - If return_scores=False: Boolean or list of booleans (is_unsafe)
            - If return_scores=True: Dict or list of dicts with 'flagged', 'score', and 'label'
        """
        if not isinstance(image, list):
            return self._check_single_image(image, return_scores)
        
        # Process list of images
        try:
            # Prepare batch processing
            pil_images = [self._load_image(img) for img in image]
            tensors = [self.classifier.preprocess(img).unsqueeze(0) for img in pil_images]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            
            with torch.no_grad():
                # Get similarity scores for all images
                batch_similarity = self.classifier(batch_tensor)
                
                # Process results
                results = []
                
                # Handle both single image and batch cases properly
                if len(image) == 1:
                    # Single image case but in a list
                    safe_score = batch_similarity[0].item()
                    unsafe_score = batch_similarity[1].item()
                    
                    # Apply softmax for probability-like scores
                    scores = torch.nn.functional.softmax(torch.tensor([safe_score, unsafe_score]), dim=0)
                    safe_prob = scores[0].item()
                    unsafe_prob = scores[1].item()
                    
                    is_unsafe = unsafe_prob > safe_prob
                    
                    if return_scores:
                        results.append({
                            'flagged': is_unsafe,
                            'score': unsafe_prob,
                            'label': 'nsfw' if is_unsafe else 'sfw'
                        })
                    else:
                        results.append(is_unsafe)
                else:
                    # Multiple images case
                    # batch_similarity shape: [batch_size, 2]
                    
                    # Apply softmax along dim=1 for each image
                    probs = torch.nn.functional.softmax(batch_similarity, dim=1)
                    
                    # Get unsafe probabilities and binary predictions
                    unsafe_probs = probs[:, 1].cpu().tolist()
                    is_unsafe = (probs[:, 1] > probs[:, 0]).cpu().tolist()
                    
                    if return_scores:
                        for i in range(len(image)):
                            results.append({
                                'flagged': is_unsafe[i],
                                'score': unsafe_probs[i],
                                'label': 'nsfw' if is_unsafe[i] else 'sfw'
                            })
                    else:
                        results = is_unsafe
            
            return results
            
        except Exception as e:
            print(f"Batch image check error: {str(e)}")
            # Fall back to processing individually if batch fails
            return [self._check_single_image(img, return_scores) for img in image]


class FinetunedQ16Detector(Q16Detector):
    """Finetuned version of Q16 detector with text-detector-like interface."""
    
    def __init__(self, device='cuda', model_name="ViT-L-14", pretrained='openai'):
        """Initialize Q16 detector with finetuned weights."""
        super().__init__(device, model_name, pretrained)
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/finetuned_q16/prompts.pt")
        self._initialize_model()
        
    def _load_prompts(self):
        """Load prompts from pickle or pt file."""
        if self.checkpoint_path.endswith("p"):
            with open(self.checkpoint_path, 'rb') as f:
                return torch.FloatTensor(pickle.load(f)).to(self.device)
        elif self.checkpoint_path.endswith("pt"):
            return torch.load(self.checkpoint_path).to(self.device).to(torch.float32)
        
        
        
        
if __name__ == "__main__":
    # Initialize checker
    checker = Q16Detector()
    
    result = checker.check("unsafe.png", return_scores=True)
    print(result)
    
    
    images = ["unsafe.png"]*100
    result = checker.check(images, return_scores=True)
    print(result)
    
