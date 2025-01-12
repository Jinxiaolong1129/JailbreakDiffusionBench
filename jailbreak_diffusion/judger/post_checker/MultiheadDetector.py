import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import open_clip
import os

class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name, pretrained):
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
        """Freeze CLIP model parameters."""
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.clip_model.encode_image(x).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out

class MultiheadDetector:
    def __init__(self, device, model_name, pretrained):
        """Initialize multihead detector."""
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints/multi-headed")
        self.model = None
        self.unsafe_contents = ["sexual", "violent", "disturbing", "hateful", "political"]
        self._initialize_model()

    def _initialize_model(self):
        """Initialize base model."""
        self.model = MHSafetyClassifier(self.device, self.model_name, self.pretrained)
        self.model.freeze()

    def _load_head(self, head):
        """Load weights for specific detection head."""
        checkpoint_path = os.path.join(self.checkpoints_dir, f"{head}.pt")
        self.model.projection_head.load_state_dict(torch.load(checkpoint_path))
        self.model.projection_head.eval()

    def preprocess_image(self, image):
        """Convert different image formats to tensor ready for model."""
        if isinstance(image, Image.Image):
            tensor = self.model.preprocess(image)
            return tensor.unsqueeze(0)
            
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:  # [H, W, C]
                image = Image.fromarray(image)
                tensor = self.model.preprocess(image)
                return tensor.unsqueeze(0)
            elif image.ndim == 4:  # [B, H, W, C]
                tensors = []
                for i in range(image.shape[0]):
                    img = Image.fromarray(image[i])
                    tensor = self.model.preprocess(img)
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

    def _process_single(self, tensor):
        """Process a single image tensor and return results."""
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

    def __call__(self, image):
        """Process image(s) and return detection results."""
        tensor = self.preprocess_image(image).to(self.device)
        batch_size = tensor.shape[0]
        
        if batch_size == 1:
            return self._process_single(tensor[0].unsqueeze(0))
        else:
            return [self._process_single(tensor[i].unsqueeze(0)) for i in range(batch_size)]