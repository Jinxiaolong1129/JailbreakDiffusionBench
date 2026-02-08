# diffusion_model/core/outputs.py

from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from PIL import Image
import torch

@dataclass
class GenerationInput:
    """Unified input structure for all generation models"""
    prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    input_images: Optional[List[Image.Image]] = None
    mask_images: Optional[List[Image.Image]] = None
    image_strength: Optional[float] = 0.8
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    
    # Video-specific parameters
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    motion_bucket_id: Optional[int] = None
    
    # Additional parameters as dict
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert input to dictionary, excluding None values"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class GenerationOutput:
    """Unified output structure for generation results"""
    images: Optional[List[Image.Image]] = None
    videos: Optional[List[torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None
    generation_time: Optional[float] = None
    
    def merge(self, other: 'GenerationOutput') -> 'GenerationOutput':
        """Merge another output into this one"""
        if other.images:
            self.images = (self.images or []) + other.images
        if other.videos:
            self.videos = (self.videos or []) + other.videos
        if other.metadata:
            self.metadata = {**(self.metadata or {}), **other.metadata}
        return self
    
    
    
    
    
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

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

    
   
# diffusion_model/api/stability.py
import base64
import requests
from PIL import Image
import io

class StabilityModel:
    """Stability AI API implementation for v2beta"""

    MODEL_ENDPOINTS = {
        "stability-ai-core": "core",
        "stability-ai-ultra": "ultra"
    }
    API_BASE = "https://api.stability.ai/v2beta/stable-image/generate"

    def __init__(self, api_key: str, model_name: str = "stability-ai-core"):
        """
        Initialize the Stability AI client.
        
        Args:
            api_key: Stability AI API key
            model_name: Model to use ('stability-ai-core' or 'stability-ai-ultra')
        """
        if model_name not in self.MODEL_ENDPOINTS:
            raise ValueError(f"Invalid model name. Must be one of: {list(self.MODEL_ENDPOINTS.keys())}")
            
        self.api_key = api_key
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        output_format: str = "png",
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        style_preset: Optional[str] = None,
        **kwargs
    ) -> GenerationOutput:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            output_format: Output format ('webp' or 'png')
            seed: Random seed for generation
            steps: Number of diffusion steps
            cfg_scale: Classifier free guidance scale
            style_preset: Style preset to use
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            GenerationOutput containing generated images and metadata
        """
        endpoint = f"{self.API_BASE}/{self.MODEL_ENDPOINTS[self.model_name]}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/*"
        }

        payload = {
            "prompt": prompt,
            "output_format": output_format
        }

        # Add optional parameters if provided
        if seed is not None:
            payload["seed"] = seed
        if steps is not None:
            payload["steps"] = steps
        if cfg_scale is not None:
            payload["cfg_scale"] = cfg_scale
        if style_preset is not None:
            payload["style_preset"] = style_preset

        # Add any additional kwargs to payload
        payload.update(kwargs)

        response = requests.post(
            endpoint,
            headers=headers,
            files={"none": ""},  # Required by the API
            data=payload
        )

        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")

        # Process the response
        image = Image.open(io.BytesIO(response.content))
        
        return GenerationOutput(
            images=[image],
            metadata={
                "model": self.model_name,
                "parameters": payload,
                "response_headers": dict(response.headers)
            }
        )

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerationOutput]:
        """
        Generate multiple images from a list of prompts.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional parameters passed to generate()
            
        Returns:
            List of GenerationOutput objects
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    
    

# test_diffusion_api.py
import os
from PIL import Image
import time

def test_model(model, prompts, save_path="test_outputs"):
    """
    测试模型生成图片
    
    Args:
        model: DallE 或 Stability 模型实例
        prompts: 要测试的 prompt 列表
        save_path: 保存生成图片的路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n开始测试 {model.__class__.__name__}")
    print("-" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f"\n测试 Prompt {i}: {prompt}")
            
            # 生成图片
            start_time = time.time()
            result = model.generate(prompt)
            end_time = time.time()
            
            # 检查结果
            if result.images and len(result.images) > 0:
                print("✓ 生成成功!")
                print(f"生成时间: {end_time - start_time:.2f}秒")
                
                # 保存图片
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                for j, img in enumerate(result.images):
                    img_path = os.path.join(save_path, f"{model.__class__.__name__}_{timestamp}_{j}.png")
                    if isinstance(img, str):  # base64 数据
                        # 处理 base64 数据并保存
                        import base64
                        img_data = base64.b64decode(img)
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                    else:  # PIL Image
                        img.save(img_path)
                    print(f"图片已保存: {img_path}")
            else:
                print("✗ 生成失败: 没有返回图片")
                
        except Exception as e:
            print(f"✗ 错误: {str(e)}")
            
    print("\n测试完成!")

if __name__ == "__main__":
    # 设置 API key
    dalle_key = ""
    stability_key = ""
    
    # 创建模型实例
    dalle_model = DallEModel(api_key=dalle_key)
    stability_model = StabilityModel(api_key=stability_key)
    
    # 测试用的 prompts
    test_prompts = [
        "A cute cat playing with a ball of yarn, digital art style",
        "A futuristic city skyline at sunset",
    ]
    
    # 测试 DALL-E
    # test_model(dalle_model, test_prompts)
    
    # 测试 Stability
    test_model(stability_model, test_prompts)