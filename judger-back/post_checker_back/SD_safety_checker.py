import torch
import numpy as np
from PIL import Image
import PIL
from diffusers.pipelines.stable_diffusion.safety_detector import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

class SD_SafetyCheckerDetector:
    def __init__(self, device):
        """Initialize Safety Checker detector."""
        self.device = device
        self.safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(self.safety_model_id)
        self.safety_detector = StableDiffusionSafetyChecker.from_pretrained(self.safety_model_id).to(self.device)
        self.safety_detector.eval()

    @staticmethod
    def numpy_to_pil(images):
        """Convert a numpy image or a batch of images to PIL images."""
        if images.ndim == 3:
            images = images[None, ...]
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def check_safety(self, x_image): # [1, H, W, C]
        """Check if image contains NSFW content."""
        safety_detector_input = self.safety_feature_extractor(
            self.numpy_to_pil(x_image), 
            return_tensors="pt"
        )
        safety_detector_input = safety_detector_input.to(self.device)
        x_checked_image, has_nsfw_concept = self.safety_detector(
            clip_input=safety_detector_input.pixel_values, 
            images=x_image
        )
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        return has_nsfw_concept

    def preprocess_image(self, image):
        """Convert different image formats to numpy array."""
        if isinstance(image, Image.Image):
            return np.expand_dims(np.array(image), axis=0)
            
        elif isinstance(image, np.ndarray):
            if image.ndim == 3:  # [H, W, C]
                return np.expand_dims(image, axis=0)
            elif image.ndim == 4:  # [B, H, W, C]
                return image
                
        elif isinstance(image, torch.Tensor):
            # Convert tensor to numpy and transpose if needed
            if image.ndim == 3:  # [C, H, W]
                image = image.permute(1, 2, 0)  # to [H, W, C]
                return np.expand_dims(image.cpu().numpy(), axis=0)
            elif image.ndim == 4:  # [B, C, H, W]
                image = image.permute(0, 2, 3, 1)  # to [B, H, W, C]
                return image.cpu().numpy()
                
        raise TypeError(
            "Input must be PIL Image, numpy array [H,W,C] or [B,H,W,C], "
            "or torch tensor [C,H,W] or [B,C,H,W]"
        )

    def __call__(self, image):
        """Process image(s) and return safety check result(s)."""
        img_array = self.preprocess_image(image)
        batch_size = img_array.shape[0]
        
        if batch_size == 1:
            has_nsfw = self.check_safety(img_array)[0]
            return bool(has_nsfw)
        else:
            results = []
            for i in range(batch_size):
                has_nsfw = self.check_safety(img_array[i:i+1])[0]
                results.append(bool(has_nsfw))
            return results