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
    def __init__(self, device, model_name, pretrained):
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

    def preprocess_image(self, image):
        """Convert different image formats to tensor ready for model."""
        if isinstance(image, Image.Image):
            tensor = self.classifier.preprocess(image)
            return tensor.unsqueeze(0)
            
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:  # [H, W, C]
                image = Image.fromarray(image)
                tensor = self.classifier.preprocess(image)
                return tensor.unsqueeze(0)
            elif image.ndim == 4:  # [B, H, W, C]
                tensors = []
                for i in range(image.shape[0]):
                    img = Image.fromarray(image[i])
                    tensor = self.classifier.preprocess(img)
                    tensors.append(tensor)
                return torch.stack(tensors)
                
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:  # [C, H, W]
                return image.unsqueeze(0)
            elif image.ndim == 4:  # [B, C, H, W]
                return image
                
        raise TypeError(
            "Input must be PIL Image, numpy array [H,W,C] or [B,H,W,C], "
            "or torch tensor [C,H,W] or [B,C,H,W]"
        )

    def __call__(self, image):
        """Process image(s) and return classification result(s)."""
        tensor = self.preprocess_image(image).to(self.device)
        y = self.classifier(tensor)
        y = torch.argmax(y, dim=1).detach().cpu().numpy()
        
        if len(y) == 1:
            return int(y[0])
        return [int(i) for i in y]

class FinetunedQ16Detector(Q16Detector):
    """Finetuned version of Q16 detector"""
    def __init__(self, device, model_name, pretrained):
        """Initialize Q16 detector with finetuned weights"""
        super().__init__(device, model_name, pretrained)
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/finetuned_q16/prompts.pt")
        self._initialize_model()

    def _load_prompts(self):
        """Load prompts from PT file."""
        return torch.load(self.checkpoint_path).to(self.device).to(torch.float32)