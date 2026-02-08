import torch
import numpy as np
from PIL import Image
from typing import Union, List, Dict, Any
from pathlib import Path
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel, AutoFeatureExtractor



from .base import BaseImageChecker


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = torch.nn.functional.normalize(image_embeds)
    normalized_text_embeds = torch.nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class IntegratedSDSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = torch.nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = torch.nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = torch.nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = torch.nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": [], "max_score": 0.0}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                special_score = concept_cos - concept_threshold + adjustment
                result_img["special_scores"][concept_idx] = round(special_score, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append((concept_idx, result_img["special_scores"][concept_idx]))
                    adjustment = 0.01
                    if special_score > result_img["max_score"]:
                        result_img["max_score"] = special_score

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                concept_score = concept_cos - concept_threshold + adjustment
                result_img["concept_scores"][concept_idx] = round(concept_score, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)
                    if concept_score > result_img["max_score"]:
                        result_img["max_score"] = concept_score

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]
        max_scores = [res["max_score"] for res in result]

        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if torch.is_tensor(images) or torch.is_tensor(images[0]):
                    images[idx] = torch.zeros_like(images[idx])  # black image
                else:
                    images[idx] = np.zeros(images[idx].shape)  # black image

        return images, has_nsfw_concepts, max_scores


class SD_SafetyCheckerDetector(BaseImageChecker):
    """Stable Diffusion safety checker for detecting NSFW content in images."""
    
    def __init__(self, device="cuda"):
        super(SD_SafetyCheckerDetector, self).__init__()
        self.device = device
        self.safety_model_id = "CompVis/stable-diffusion-safety-checker"
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(self.safety_model_id)
        
        # Load the IntegratedSDSafetyChecker instead of the original StableDiffusionSafetyChecker
        model = IntegratedSDSafetyChecker.from_pretrained(self.safety_model_id).to(self.device)
        model.eval()
        self.safety_detector = model

    def check(self, image_input: Union[str, Path, Image.Image, 
                                     List[Union[str, Path, Image.Image]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Unified check method for all detectors.
        Returns a dictionary containing:
        - flagged: True if unsafe content detected, False otherwise
        - score: The maximum score for NSFW content
        - label: 'nsfw' if flagged, 'sfw' otherwise
        """
        return self.__call__(image_input)

    def __call__(self, image_input: Union[str, Path, Image.Image, 
                                        List[Union[str, Path, Image.Image]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process image(s) and return detailed check result(s).
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
        """Process a batch of PIL images and return detailed safety information."""
        # Extract features for all images
        image_features = self.safety_feature_extractor(images, return_tensors="pt")
        image_array = image_features.pixel_values.to(self.device)
        
        # Batch size
        batch_size = image_array.shape[0]
        
        # Check safety for all images in batch
        with torch.no_grad():
            # Make a copy of the image features for potential modification by the model
            image_copies = image_array.clone()
            
            # Run safety check with modified detector that returns max scores
            _, has_nsfw_concepts, max_scores = self.safety_detector(image_array, image_copies)
            
        # Convert to detailed result dictionaries
        return [
            {
                'flagged': bool(has_nsfw),
                'score': float(max_score),
                'label': 'nsfw' if has_nsfw else 'sfw'
            }
            for has_nsfw, max_score in zip(has_nsfw_concepts, max_scores)
        ]

    def _process_single_image(self, image):
        """Process a single PIL image and return detailed safety information."""
        # Add batch dimension
        images = [image]
        
        # Extract features
        image_features = self.safety_feature_extractor(images, return_tensors="pt")
        image_array = image_features.pixel_values.to(self.device)
        
        # Check safety
        with torch.no_grad():
            image_copy = image_array.clone()
            _, has_nsfw_concepts, max_scores = self.safety_detector(image_array, image_copy)
            
        # Return detailed result as dictionary
        return {
            'flagged': bool(has_nsfw_concepts[0]),
            'score': float(max_scores[0]),
            'label': 'nsfw' if has_nsfw_concepts[0] else 'sfw'
        }

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
    
    
if __name__ == "__main__":
    # Initialize checker
    checker = SD_SafetyCheckerDetector()
    
    # Check a single image
    result = checker.check("unsafe.png")
    print(f"Image flagged: {result['flagged']}, Score: {result['score']:.4f}, Label: {result['label']}")
    
    # Example for batch processing
    batch_results = checker.check(["unsafe.png"]*100)
    for i, res in enumerate(batch_results):
        print(f"Image {i+1} - Flagged: {res['flagged']}, Score: {res['score']:.4f}, Label: {res['label']}")