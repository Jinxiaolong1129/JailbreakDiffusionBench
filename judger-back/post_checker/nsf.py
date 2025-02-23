import os
import torch
import numpy as np
from PIL import Image
from typing import Union
import clip

# 目前版本不支持

class ClipSafetyModel:
    """
    A class for CLIP-based NSFW detection.
    """
    def __init__(self, model_name: str = "ViT-L/14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.dim = 768 if model_name == "ViT-L/14" else None

        if self.dim is None:
            raise ValueError(f"Unsupported model: {model_name}")

    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Process the image and generate embeddings.

        Args:
            image (Union[str, Image.Image]): Path to the image or PIL Image instance.

        Returns:
            np.ndarray: Normalized image embeddings.
        """
        if isinstance(image, str):
            image = Image.open(image)

        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)

        return self._normalize(image_features.cpu().numpy())

    @staticmethod
    def _normalize(features: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Normalize the input embeddings to unit norm.

        Args:
            features (np.ndarray): Input features.
            axis (int): Axis along which to normalize.

        Returns:
            np.ndarray: Normalized features.
        """
        norm = np.linalg.norm(features, axis=axis, keepdims=True)
        norm[norm == 0] = 1
        return features / norm


class NSF_Checker:
    """
    Safety checker for NSFW prediction.
    """
    def __init__(self, clip_model: str = "ViT-L/14"):
        self.safety_model = self._load_safety_model()
        self.clip_model = ClipSafetyModel(model_name=clip_model)

    @staticmethod
    def _load_safety_model():
        """
        Load the NSFW safety model.

        Returns:
            TensorFlow/Keras model instance.
        """
        from tensorflow.keras.models import load_model
        import autokeras as ak

        cache_folder = "./NSFW-cache"
        model_dir = os.path.join(cache_folder, "clip_autokeras_binary_nsfw")

        if not os.path.exists(model_dir):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"

            from urllib.request import urlretrieve
            import zipfile

            zip_path = os.path.join(cache_folder, "clip_autokeras_binary_nsfw.zip")
            urlretrieve(url_model, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_folder)

        return load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)

    def predict_nsfw(self, image: Union[str, Image.Image]) -> float:
        """
        Predict the NSFW score for the given image.

        Args:
            image (Union[str, Image.Image]): Path to the image or PIL Image instance.

        Returns:
            float: NSFW score (probability).
        """
        image_embedding = self.clip_model.encode_image(image)
        nsfw_score = self.safety_model.predict(image_embedding)
        return nsfw_score[0][0]

# Example usage:
nsfw_detector = NSF_Checker()
score = nsfw_detector.predict_nsfw("unsafe.png")
print("NSFW Score:", score)
