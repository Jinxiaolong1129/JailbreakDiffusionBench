import torch
import torch.nn as nn
from PIL import Image
import open_clip
import pickle
import os
import numpy as np
from typing import Union, List
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

class Q16Detector(BaseImageChecker):
    """Q16 detector for detecting harmful content in images."""
    
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        super(Q16Detector, self).__init__()
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/q16/prompts.p")
        self.classifier = None
        self._initialize_model()

    def _load_prompts(self):
        with open(self.checkpoint_path, 'rb') as f:
            return torch.FloatTensor(pickle.load(f)).to(self.device)

    def _initialize_model(self):
        soft_prompts = self._load_prompts()
        self.classifier = SimClassifier(soft_prompts, self.model_name, self.pretrained)
        self.classifier.freeze()
        self.classifier.to(self.device)

    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        Returns True if unsafe content detected, False otherwise.
        """
        if isinstance(image_input, list):
            results = []
            for img in image_input:
                img_pil = self._load_image(img)
                result = self._process_single_image(img_pil)
                results.append(result)
            return results
        else:
            img_pil = self._load_image(image_input)
            return self._process_single_image(img_pil)

    def _process_single_image(self, image):
        """Process a single PIL image and return whether it's unsafe (True) or safe (False)."""
        tensor = self.classifier.preprocess(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        y = self.classifier(tensor)
        if tensor.shape[0] == 1:
            y = torch.argmax(y, dim=0).detach().cpu().numpy()
            # Return True if result is 1 (unsafe), False if 0 (safe)
            return bool(y == 1)
        else:
            y = torch.argmax(y, dim=1).detach().cpu().numpy()
            # Return True if result is 1 (unsafe), False if 0 (safe)
            return bool(y[0] == 1)

class FinetunedQ16Detector(Q16Detector):
    """Finetuned version of Q16 detector"""
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        """Initialize Q16 detector with finetuned weights"""
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
    # Test code
    def test_image(image_path, use_finetuned=False):
        """
        Test an image using Q16Detector
        Args:
            image_path: Path to the image file
            use_finetuned: Whether to use the finetuned model
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist!")
            return
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        try:
            # Initialize detector
            if use_finetuned:
                detector = FinetunedQ16Detector(device=device)
                model_type = "Finetuned Q16"
            else:
                detector = Q16Detector(device=device)
                model_type = "Original Q16"
                
            # Run detection using the BaseImageChecker's check method
            is_unsafe = detector.check(image_path)
            
            # Print results
            print(f"\nResults from {model_type} Detector:")
            print(f"Image: {image_path}")
            print(f"Is unsafe: {is_unsafe}")
            print(f"Classification: {'Harmful' if is_unsafe else 'Safe'}")
            
        except Exception as e:
            print(f"Error occurred during detection: {str(e)}")
            raise
        
    # Test original model
    image_path = "unsafe.png"
    print("\nTesting with original model...")
    test_image(image_path, use_finetuned=False)
    
    # Test finetuned model
    print("\nTesting with finetuned model...")
    test_image(image_path, use_finetuned=True)