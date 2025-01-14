import torch
import torch.nn as nn
import PIL
from PIL import Image
import open_clip
import pickle
import os
import numpy as np

class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, model_name, pretrained):
        super(SimClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(torch.float32)
        self.prompts = torch.nn.Parameter(embeddings)
        
    def freeze(self):
        """Freeze CLIP model parameters."""
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_feat @ text_feat.T)
        return similarity.squeeze()

class Q16Detector:
    """Original Q16 model detector"""
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        """Initialize Q16 detector with original weights"""
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/q16/prompts.p")
        self.classifier = None
        self._initialize_model()

    def _load_prompts(self):
        """Load prompts from pickle file."""
        with open(self.checkpoint_path, 'rb') as f:
            return torch.FloatTensor(pickle.load(f)).to(self.device)

    def _initialize_model(self):
        """Initialize and prepare the model."""
        soft_prompts = self._load_prompts()
        self.classifier = SimClassifier(soft_prompts, self.model_name, self.pretrained)
        self.classifier.freeze()
        self.classifier.to(self.device)

    def __call__(self, image):
        """Process image and return classification result."""
        tensor = self.classifier.preprocess(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        y = self.classifier(tensor)
        y = torch.argmax(y, dim=0).detach().cpu().numpy()
        return int(y)
        # return int(y[0])

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