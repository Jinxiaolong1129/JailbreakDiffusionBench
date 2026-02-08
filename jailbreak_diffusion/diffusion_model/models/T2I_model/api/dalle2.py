from openai import OpenAI
from openai import BadRequestError
from PIL import Image
import requests
from io import BytesIO
from ....core.outputs import GenerationOutput
import logging
import os

class DallE2Model:
    """DALL-E API implementation using the OpenAI SDK with safety handling"""

    def __init__(self, api_key: str, model_name: str = "dall-e-3"):
        """
        Initialize the DALL-E model API wrapper.

        Args:
            api_key (str): API key for authentication.
            model_name (str): Name of the DALL-E model to use. Default is "dall-e-3".
        """
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def generate(self, input_data, **kwargs) -> GenerationOutput:
        """
        Generate images based on the given prompts and additional parameters.
        Handles content policy violations by returning None for images.

        Args:
            input_data: An object containing prompts and negative_prompt.
            **kwargs: Optional parameters such as:
                - n (int): Number of images to generate. Default is 1.
                - size (str): Size of the images (e.g., "1024x1024"). Default is "1024x1024".
                - quality (str): Quality level of the images ("standard" or "hd"). Default is "standard".
                - response_format (str): Format of the response ("url" or "b64_json"). Default is "b64_json".
                - style (str): Style of the image ("vivid" or "natural"). Default is None.

        Returns:
            GenerationOutput: An object containing the generated images and metadata.
        """
        params = input_data.extra_params or {}
        all_images = []
        metadata = {
            "model": self.model_name,
            "parameters": params,
            "errors": []
        }

        payload = {
            "model": self.model_name,
            "n": params.get("n", 1),
            "size": params.get("size", "1024x1024"),
            "quality": params.get("quality", "standard"),
            "response_format": params.get("response_format", "url"),
        }

        if "style" in params:
            payload["style"] = params["style"]

        for prompt in input_data.prompts:
            try:
                payload["prompt"] = prompt
                
                # Handle negative prompt if provided
                if input_data.negative_prompt and len(input_data.negative_prompt) > 0:
                    # DALL-E doesn't directly support negative prompts, but we can adapt it
                    # by appending to the prompt with instructions to avoid certain elements
                    payload["prompt"] += f". Please avoid: {input_data.negative_prompt[0]}"
                
                response = self.client.images.generate(**payload)
                
                images = self._process_response(response, payload["response_format"])
                all_images.extend(images)
                
            except BadRequestError as e:
                # Log the error
                self.logger.warning(f"Content policy violation detected for prompt: {prompt}")
                self.logger.warning(f"Error details: {str(e)}")
                
                # Check if it's a content policy violation
                if (hasattr(e, 'response') and 
                    getattr(e.response, 'status_code', None) == 400 and 
                    'content_policy_violation' in str(e)):
                    # Add None to the images list for this prompt
                    all_images.append(None)
                    metadata["errors"].append({
                        "prompt": prompt,
                        "error": "content_policy_violation",
                        "message": str(e)
                    })
                else:
                    # Re-raise if it's not a content policy violation
                    raise

        return GenerationOutput(
            images=all_images,
            metadata=metadata
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
                img_response = requests.get(item.url)
                img_response.raise_for_status()
                images.append(Image.open(BytesIO(img_response.content)))
            elif response_format == "b64_json":
                images.append(item.b64_json)
            else:
                raise ValueError("Unsupported response format: " + response_format)

        return images