import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import open_clip
import os
from typing import Union, List, Optional
from pathlib import Path
from .base import BaseImageChecker

class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(device)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
        ).to(device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, image):
        x = self.clip_model.encode_image(image).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out

class MultiheadDetector(BaseImageChecker):
    """Multi-headed safety classifier for detecting potentially problematic image content."""
    
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        super(MultiheadDetector, self).__init__()
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints/multi-headed")
        self.model = None
        self.unsafe_contents = ["sexual", "violent", "disturbing", "hateful", "political"]
        self._initialize_model()

    def _initialize_model(self):
        self.model = MHSafetyClassifier(self.device, self.model_name, self.pretrained)
        self.model.freeze()

    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        Returns True if unsafe content detected.
        """
        if isinstance(image_input, list):
            results = []
            for img in image_input:
                img_pil = self._load_image(img)
                result = self._process_single_image(img_pil)
                results.append(result['unsafe'])
            return results
        else:
            img_pil = self._load_image(image_input)
            result = self._process_single_image(img_pil)
            return result['unsafe']
    
    def _process_single_image(self, image):
        """Process a single PIL image and return detailed results."""
        tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
        return self._process_single(tensor)

    def _process_single(self, tensor):
        """Process a single tensor and return detailed results."""
        results = {}
        with torch.no_grad():
            head_results = []
            for head in self.unsafe_contents:
                self._load_head(head)
                logits = self.model(tensor).squeeze()
                pred = (logits.detach().cpu() > 0.5).to(dtype=torch.int64)
                results[head] = bool(pred.item())
                head_results.append(pred.item())
                
        results['unsafe'] = bool(sum(head_results) > 0)
        return results

    def _load_head(self, head):
        """Load the weights for a specific detection head."""
        checkpoint_path = os.path.join(self.checkpoints_dir, f"{head}.pt")
        self.model.projection_head.load_state_dict(torch.load(checkpoint_path))
        self.model.projection_head.eval()
        
    def get_detailed_results(self, image_input: Union[str, Path, Image.Image]) -> dict:
        """
        Get detailed results with scores for each category.
        This provides more information than the standard check method.
        
        Args:
            image_input: Image to check
            
        Returns:
            Dictionary with results for each unsafe content category and overall unsafe status
        """
        img_pil = self._load_image(image_input)
        tensor = self.model.preprocess(img_pil).unsqueeze(0).to(self.device)
        return self._process_single(tensor)


if __name__ == "__main__":
    def test_image(image_path, model_name="ViT-L-14", pretrained="openai"):
        """
        Test an image using MultiheadDetector
        Args:
            image_path: Path to the image file
            model_name: Name of the CLIP model to use
            pretrained: Name of the pretrained weights
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
            detector = MultiheadDetector(
                device=device,
                model_name=model_name,
                pretrained=pretrained
            )
            
            # Run detection using the BaseImageChecker's check method
            is_unsafe = detector.check(image_path)
            
            # Get detailed results
            details = detector.get_detailed_results(image_path)
            
            # Print results
            print("\nMultihead Detection Results:")
            print(f"Image: {image_path}")
            print("\nDetection Results by Category:")
            for category, is_detected in details.items():
                if category != 'unsafe':
                    print(f"{category.capitalize()}: {'Detected' if is_detected else 'Not Detected'}")
            
            print(f"\nOverall Safety: {'Unsafe' if is_unsafe else 'Safe'}")
            
        except Exception as e:
            print(f"Error occurred during detection: {str(e)}")
            raise

    # Test single image
    print("\nTesting single image...")
    test_image("unsafe.png")