from typing import Union, List, Optional, Dict
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Default policy prompt moved outside the class
DEFAULT_POLICY = """
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content: 
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}
"""


class LlavaGuardChecker:
    """LlavaGuard image content checker for safety assessment of local images"""
    
    def __init__(self, model_name='AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf', device='cuda:0'):
        """
        Initialize the LlavaGuard checker.
        
        Args:
            model_name: Name or path of the LlavaGuard model
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
        """
        # Load model and processor
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Set device
        self.device = device
        
        # Generation parameters
        self.hyperparameters = {
            "max_new_tokens": 200,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "num_beams": 2,
            "use_cache": True,
        }

    def _load_image(self, image_source: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load an image from a local file path or return the Image object if already loaded.
        
        Args:
            image_source: Path to the local image file or a PIL Image object
            
        Returns:
            PIL Image object
        """
        if isinstance(image_source, Image.Image):
            return image_source
        
        try:
            return Image.open(str(image_source))
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_source}: {str(e)}")

    def check_image(
        self,
        image_source: Union[str, Path, Image.Image],
        policy: str = DEFAULT_POLICY
    ) -> Dict[str, Union[bool, str]]:
        """
        Check a single local image or PIL Image object using LlavaGuard.
        
        Args:
            image_source: Path to the local image file or a PIL Image object
            policy: Policy prompt to use for assessment (default is DEFAULT_POLICY)
            
        Returns:
            Dict containing:
                - flagged: bool indicating if unsafe content detected
                - response: str containing model's explanation
        """
        # Load and process image
        img = self._load_image(image_source)
        
        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": policy},
                ],
            },
        ]
        
        # Process input
        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_prompt,
            images=img,
            return_tensors="pt"
        )
        
        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(**inputs, **self.hyperparameters)
        
        # Decode response
        prompt_len = inputs['input_ids'].shape[-1]
        response = self.processor.decode(output[0][prompt_len:], skip_special_tokens=True)
        
        # Parse result
        is_unsafe = 'unsafe' in response.lower()
        
        return {
            "flagged": is_unsafe,
            "response": response
        }

    def check(
        self,
        image_sources: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        policy: str = DEFAULT_POLICY
    ) -> Union[Dict[str, Union[bool, str]], List[Dict[str, Union[bool, str]]]]:
        """
        Process one or more local images or PIL Image objects and return check result(s).
        
        Args:
            image_sources: Path(s) to local image file(s) or PIL Image object(s)
            policy: Policy prompt to use for assessment (default is DEFAULT_POLICY)
            
        Returns:
            Dict or list of dicts containing check results
        """
        if isinstance(image_sources, list):
            results = [
                self.check_image(img_source, policy)
                for img_source in image_sources
            ]
            return results
        else:
            return self.check_image(image_sources, policy)

    def get_explanation(
        self,
        image_path: Union[str, Path],
        policy: str = DEFAULT_POLICY
    ) -> str:
        """
        Get detailed explanation for a local image check.
        
        Args:
            image_path: Path to the local image file
            policy: Policy prompt to use for assessment (default is DEFAULT_POLICY)
            
        Returns:
            String containing model's explanation
        """
        result = self.check_image(image_path, policy)
        return result["explanation"]


# Example usage
if __name__ == "__main__":
    # Initialize checker
    checker = LlavaGuardChecker(device='cuda:0')
    
    # Check a single image
    image_path = "unsafe.png"
    
    # You can use the default policy
    result = checker.check(image_path)
    print(result)
    print('='*50)
    
    image_paths = [
        "unsafe.png",
        "sneak.png",
    ]
    try:
        results = checker.check(image_paths)
        print(results)
        print('='*50)
        

    except Exception as e:
        print(f"Check failed: {str(e)}")