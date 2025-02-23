import os
import numpy as np
import torch
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import autokeras as ak
from urllib.request import urlretrieve
import zipfile
import open_clip



class ClipNSFWDetector:
    """CLIP-based NSFW content detector using AutoKeras model."""
    
    def __init__(self, device, clip_model="ViT-L/14"):
        """Initialize CLIP NSFW detector.
        
        Args:
            device: torch device
            clip_model: CLIP model variant (currently only supports 'ViT-L/14')
        """
        self.device = device
        self.clip_model = clip_model
        self.model_cache = os.path.join(os.path.dirname(__file__), "checkpoints/clip_nsfw")
        
        # Initialize CLIP model
        self.clip, self.preprocess, _ = open_clip.create_model_and_transforms(clip_model, pretrained="openai")
        self.clip.to(device)
        self.clip.eval()
        
        # Load AutoKeras NSFW detector
        self.detector = self._load_nsfw_detector()

    def _load_nsfw_detector(self):
        """Download and load the AutoKeras NSFW detection model."""
        if self.clip_model == "ViT-L/14":
            model_dir = os.path.join(self.model_cache, "clip_autokeras_binary_nsfw")
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        else:
            raise ValueError(f"Unsupported CLIP model: {self.clip_model}")

        # Create cache directory if it doesn't exist
        os.makedirs(self.model_cache, exist_ok=True)
        
        # Download and extract if not already present
        if not os.path.exists(model_dir):
            zip_path = os.path.join(self.model_cache, "clip_autokeras_binary_nsfw.zip")
            urlretrieve(url_model, zip_path)
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.model_cache)
                
            # Clean up zip file
            os.remove(zip_path)
        
        # Load the model on CPU
        with tf.device('/cpu:0'):
            model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS, compile=False)
        return model

    def preprocess_image(self, image):
        """Convert different image formats to CLIP embeddings."""
        if isinstance(image, Image.Image):
            image = self.preprocess(image).unsqueeze(0)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
                image = self.preprocess(image).unsqueeze(0)
            elif image.ndim == 4:
                images = []
                for i in range(image.shape[0]):
                    img = Image.fromarray(image[i])
                    img = self.preprocess(img)
                    images.append(img)
                image = torch.stack(images)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
        else:
            raise TypeError(
                "Input must be PIL Image, numpy array [H,W,C] or [B,H,W,C], "
                "or torch tensor [C,H,W] or [B,C,H,W]"
            )
        return image.to(self.device)

    def get_clip_embeddings(self, image):
        """Get CLIP embeddings for image(s)."""
        with torch.no_grad():
            embeddings = self.clip.encode_image(image)
            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def check_nsfw(self, embeddings):
        """Check if embeddings indicate NSFW content using AutoKeras model."""
        with tf.device('/cpu:0'):
            embeddings = embeddings.cpu().numpy()
            nsfw_scores = self.detector.predict(embeddings)
            return (nsfw_scores > 0.5).astype(bool)

    def __call__(self, image):
        """Process image(s) and return NSFW detection results.
        
        Args:
            image: PIL Image, numpy array, or torch tensor
            
        Returns:
            bool or list of bool: True if NSFW content detected
        """
        # Preprocess and get embeddings
        image_tensor = self.preprocess_image(image)
        embeddings = self.get_clip_embeddings(image_tensor)
        
        # Check NSFW content
        results = self.check_nsfw(embeddings)
        
        # Return single result for single image
        if len(results) == 1:
            return bool(results[0])
        return [bool(r) for r in results]