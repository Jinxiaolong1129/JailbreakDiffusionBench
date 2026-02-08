import os
from typing import Union, List
import base64
import openai
from concurrent.futures import ThreadPoolExecutor
from .base import BaseImageChecker

class GPT_4o_mini_ImageChecker():
    """Content checker using GPT-4o-mini API for image moderation with batch processing capabilities."""
    
    def __init__(self, prompt_file: str = 'gpt_4o_mini_image_simple', model: str = "gpt-4o-mini", max_workers: int = 512):
        """Initialize with API key from environment variable, model, and max worker threads.
        
        Args:
            prompt_file: Path to prompt file for image moderation
            model: GPT vision model to use
            max_workers: Maximum number of concurrent API requests
        """
        super().__init__()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        prompt_file = f'jailbreak_diffusion/judger/post_checker/prompts/{prompt_file}.md'
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        
        # Load prompt from file if provided
        self.prompt_template = self._load_prompt_template(prompt_file)
        
    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from a file."""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading prompt file {prompt_file}: {str(e)}")
            raise ValueError(f"Could not load prompt file: {prompt_file}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API transmission."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            raise ValueError(f"Could not encode image: {image_path}")
    
    def _check_single_image(self, image_path: str) -> dict:
        """Check single image using GPT-4o-mini vision capabilities with the defined prompt template."""
        try:
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            
            # Format the prompt template
            prompt = self.prompt_template
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=50,
                temperature=1e-6,
            )
            
            return {
                "flagged": "UNSAFE" in response.choices[0].message.content,
                "response": response.choices[0].message.content
            }
            
        except Exception as e:
            print(f"GPT Vision API error: {str(e)}")
            return {"flagged": True, "response": f"Error: {str(e)}"}

    def _batch_check(self, image_paths: List[str]) -> List[dict]:
        """Check multiple images in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._check_single_image, image_paths))
            
    def check(self, image_path: Union[str, List[str]], return_scores=False) -> Union[dict, List[dict]]:
        """Check image(s) for inappropriate content.
        
        Args:
            image_path: Single image path or list of image paths to check
            return_scores: Not implemented in this version, kept for API compatibility
            
        Returns:
            Dictionary (for single image) or list of dictionaries (for multiple images)
            with "flagged" boolean indicating if content is unsafe (True) or safe (False)
            and "response" containing the model's response
        """
        if isinstance(image_path, str):
            return self._check_single_image(image_path)
        
        # Use batch processing for lists
        return self._batch_check(image_path)


# test
if __name__ == "__main__":
    # Example using default prompt
    default_checker = GPT_4o_mini_ImageChecker(max_workers=3)
    
    # Example with custom prompt files
    custom_checker = GPT_4o_mini_ImageChecker(prompt_file="gpt_4o_mini_image_detailed", max_workers=3)
    
    # Test images
    image_paths = [
        "path/to/safe_image.jpg",
        "path/to/questionable_image.jpg",
        "path/to/nsfw_image.jpg"
    ]
    
    print("Testing with default image prompt:")
    results = default_checker.check(image_paths)
    for path, result in zip(image_paths, results):
        print(f"Image: {path}")
        print(f"Unsafe: {result['flagged']}")
        print(f"Response: {result['response']}\n")