import torch
import numpy as np
from PIL import Image
import PIL
from diffusers.pipelines.stable_diffusion.safety_detector import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

class SafetyCheckerDetector:
    def __init__(self, device):
        self.device = device
        self.safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(self.safety_model_id)
        self.safety_detector = StableDiffusionSafetyChecker.from_pretrained(self.safety_model_id).to(self.device)
        self.safety_detector.eval()

    def check(self, image_input: Union[str, Path, Image.Image, 
                                     List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        Returns True if unsafe content detected.
        """
        if isinstance(image_input, list):
            processed_images = []
            for img in image_input:
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                processed_images.append(img)
            return self(processed_images)
        else:
            if isinstance(image_input, (str, Path)):
                image_input = Image.open(image_input)
            return self(image_input)

    def __call__(self, image):
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