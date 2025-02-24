import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from typing import Union, List
from pathlib import Path

class SD_SafetyCheckerDetector:
    def __init__(self, device="cuda"):
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

    def check_safety(self, image_features: torch.Tensor) -> List[bool]:
        """
        Check if the image features contain unsafe content.
        Returns a list of boolean values indicating unsafe content.
        """
        with torch.no_grad():
            images = image_features.clone()  # Make a copy for potential modification
            safety_output = self.safety_detector(image_features, images)
            has_nsfw_concepts = safety_output[1]  # Second output is has_nsfw_concepts
            return has_nsfw_concepts

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

# test
if __name__ == "__main__":
    detector = SD_SafetyCheckerDetector()
    test_image = Image.open("unsafe.png")
    result = detector.check(test_image)
    print('='*50)
    print(result)
    print(f"Image is {'unsafe' if result else 'safe'}")