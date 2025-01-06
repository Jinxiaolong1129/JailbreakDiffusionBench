from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
from core.outputs import GenerationOutput

class DallEModel:
    """DALL-E API implementation using the OpenAI SDK"""

    def __init__(self, api_key: str, model_name: str = "dall-e-3"):
        """
        Initialize the DALL-E model API wrapper.

        Args:
            api_key (str): API key for authentication.
            model_name (str): Name of the DALL-E model to use. Default is "dall-e-3".
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """
        Generate images based on the given prompt and additional parameters.

        Args:
            prompt (str): The textual description for image generation.
            **kwargs: Optional parameters such as:
                - n (int): Number of images to generate. Default is 1.
                - size (str): Size of the images (e.g., "1024x1024"). Default is "1024x1024".
                - quality (str): Quality level of the images ("standard" or "hd"). Default is "standard".
                - response_format (str): Format of the response ("url" or "b64_json"). Default is "b64_json".
                - style (str): Style of the image ("vivid" or "natural"). Default is None.

        Returns:
            GenerationOutput: An object containing the generated images and metadata.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "response_format": kwargs.get("response_format", "b64_json"),
        }

        if "style" in kwargs:
            payload["style"] = kwargs["style"]

        response = self.client.images.generate(**payload)

        images = self._process_response(response, payload["response_format"])

        return GenerationOutput(
            images=images,
            metadata={
                "model": self.model_name,
                "parameters": payload,
                "raw_response": response
            }
        )

    def _process_response(self, response, response_format):
        """
        Process the API response to extract image data.

        Args:
            response (dict): The response from the OpenAI API.
            response_format (str): Format of the response ("url" or "b64_json").

        Returns:
            list: A list of PIL.Image objects or raw Base64 data depending on the format.

        Raises:
            ValueError: If the response format is invalid.
        """
        images = []

        for item in response.data:
            if response_format == "url":
                img_response = requests.get(item["url"])
                img_response.raise_for_status()
                images.append(Image.open(BytesIO(img_response.content)))
            elif response_format == "b64_json":
                images.append(item.b64_json)
            else:
                raise ValueError("Unsupported response format: " + response_format)

        return images
