import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from typing import Union, List
from pathlib import Path
from .base import BaseImageChecker

class SD_SafetyCheckerDetector(BaseImageChecker):
    """Stable Diffusion safety checker for detecting NSFW content in images."""
    
    def __init__(self, device="cuda"):
        super(SD_SafetyCheckerDetector, self).__init__()
        self.device = device
        self.safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(self.safety_model_id)
        self.safety_detector = StableDiffusionSafetyChecker.from_pretrained(self.safety_model_id).to(self.device)
        self.safety_detector.eval()

    def check(self, image_input: Union[str, Path, Image.Image, 
                                     List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Unified check method for all detectors.
        Returns True if unsafe content detected.
        """
        return self.__call__(image_input)

    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        Returns True if unsafe content detected, False otherwise.
        """
        if isinstance(image_input, list):
            if not image_input:
                return []
                
            # Load all images as PIL images
            pil_images = [self._load_image(img) for img in image_input]
            
            # Convert to RGB if needed
            rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in pil_images]
            
            # Process all images in a batch
            return self._process_batch(rgb_images)
        else:
            img_pil = self._load_image(image_input)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            return self._process_single_image(img_pil)

    def _process_batch(self, images):
        """Process a batch of PIL images and return whether each is unsafe (True) or safe (False)."""
        # Extract features for all images
        image_features = self.safety_feature_extractor(images, return_tensors="pt")
        image_array = image_features.pixel_values.to(self.device)
        
        # Batch size
        batch_size = image_array.shape[0]
        
        # Check safety for all images in batch
        with torch.no_grad():
            # Make a copy of the image features for potential modification by the model
            image_copies = image_array.clone()
            
            # Run safety check
            _, has_nsfw_concepts = self.safety_detector(image_array, image_copies)
            
        # Convert to Python booleans
        return [bool(has_nsfw) for has_nsfw in has_nsfw_concepts]

    def _process_single_image(self, image):
        """Process a single PIL image and return whether it's unsafe (True) or safe (False)."""
        # Add batch dimension
        images = [image]
        
        # Extract features
        image_features = self.safety_feature_extractor(images, return_tensors="pt")
        image_array = image_features.pixel_values.to(self.device)
        
        # Check safety
        with torch.no_grad():
            image_copy = image_array.clone()
            _, has_nsfw_concepts = self.safety_detector(image_array, image_copy)
            
        # Return result as boolean
        return bool(has_nsfw_concepts[0])

    def preprocess_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Preprocess image(s) for the safety checker model.
        """
        if isinstance(image, list):
            if not image:
                raise ValueError("Empty image list provided")
            images = [img.convert("RGB") if img.mode != "RGB" else img for img in image]
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            images = [image]

        # Process images using the feature extractor
        image_features = self.safety_feature_extractor(images, return_tensors="pt")
        image_array = image_features.pixel_values.to(self.device)
        return image_array