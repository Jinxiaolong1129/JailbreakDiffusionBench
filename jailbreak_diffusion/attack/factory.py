# jailbreak_diffusion/attack/factory.py
from typing import Any, Dict, List, Type, Optional, Callable, Union
from .base import BaseAttacker, AttackResult
from jailbreak_diffusion.judger.pre_checker import (
    OpenAITextDetector, AzureTextDetector, GoogleTextModerator, GPTChecker, 
    LlamaGuardChecker, NSFW_text_classifier_Checker, NSFW_word_match_Checker,
    distilbert_nsfw_text_checker, distilroberta_nsfw_text_checker, 
    NvidiaAegisChecker, DetoxifyChecker
)
from jailbreak_diffusion.judger.post_checker import (
    MultiheadDetector, Q16Detector, FinetunedQ16Detector, SD_SafetyCheckerDetector,
    OpenAIImageDetector, AzureContentSafetyDetector,
    LlavaGuardChecker, GPT_4o_ImageChecker
)

class AttackerFactory:
    _registry: Dict[str, Type[BaseAttacker]] = {}
    
    def __init__(
        self, 
        attack_type: str, 
        target_model: Any,
        text_detector: Optional[Union[str, Dict, Callable[[str], bool]]] = None,
        image_detector: Optional[Union[str, Dict, Callable[[Any], bool]]] = None,
        **kwargs
    ) -> None:
        """
        Initialize an attacker with specified type and checkers.
        
        Args:
            attack_type: Name of the attack method to use
            target_model: Model to be attacked
            text_detector: Text detector name, config dict, or callable function
            image_detector: Image detector name, config dict, or callable function
            **kwargs: Additional arguments passed to attacker
        """
        if attack_type not in self._registry:
            raise ValueError(f"Attack type '{attack_type}' not found. Available: {list(self._registry.keys())}")
        
        # Initialize text detector if specified
        text_detector_instance = self._init_detector(text_detector, is_text=True)
        
        # Initialize image detector if specified
        image_detector_instance = self._init_detector(image_detector, is_text=False)
        
        attacker_class = self._registry[attack_type]
        self.attacker = attacker_class(
            target_model=target_model,
            text_detector=text_detector_instance,
            image_detector=image_detector_instance,
            **kwargs
        )
    
    def _init_detector(self, detector_config, is_text=True) -> Optional[Callable]:
        """
        Initialize a detector based on the provided configuration.
        
        Args:
            detector_config: Can be:
                - None: returns None
                - String: name of a built-in detector
                - Dict: configuration for a built-in detector
                - Callable: already initialized detector function
            is_text: Whether this is a text detector (True) or image detector (False)
            
        Returns:
            Initialized detector function or None
        """
        if detector_config is None:
            return None
        
        # If already a callable function, return it
        if callable(detector_config):
            return detector_config
        
        # Get the appropriate detector mapping based on type
        detector_mapping = self._get_text_detector_mapping() if is_text else self._get_image_detector_mapping()
        
        # Case where detector_config is a string (name of detector)
        if isinstance(detector_config, str):
            if detector_config not in detector_mapping:
                raise ValueError(f"Detector '{detector_config}' not found. Available: {list(detector_mapping.keys())}")
            return detector_mapping[detector_config]()
        
        # Case where detector_config is a dict with name and params
        elif isinstance(detector_config, dict):
            if "name" not in detector_config:
                raise ValueError("Detector config dict must contain 'name' field")
                
            name = detector_config["name"]
            if name not in detector_mapping:
                raise ValueError(f"Detector '{name}' not found. Available: {list(detector_mapping.keys())}")
                
            # Extract params excluding 'name'
            params = {k: v for k, v in detector_config.items() if k != "name"}
            return detector_mapping[name](**params)
        
        else:
            raise ValueError(f"Invalid detector configuration type: {type(detector_config)}")

    def _get_text_detector_mapping(self) -> Dict[str, Type]:
        """Get mapping of text detector names to classes"""
        return {
            "openai_text_moderation": OpenAITextDetector,
            "NSFW_text_classifier": NSFW_text_classifier_Checker,
            "NSFW_word_match": NSFW_word_match_Checker,
            "distilbert_nsfw_text_checker": distilbert_nsfw_text_checker,
            "distilroberta_nsfw_text_checker": distilroberta_nsfw_text_checker,
            "gpt_4o_mini": GPTChecker,
            "gpt_4o": GPTChecker,
            "llama_guard": LlamaGuardChecker,
            "azure_text_detector": AzureTextDetector,
            "google_text_checker": GoogleTextModerator,
            "nvidia_aegis": NvidiaAegisChecker,
            "detoxify": DetoxifyChecker
        }
    
    def _get_image_detector_mapping(self) -> Dict[str, Type]:
        """Get mapping of image detector names to classes"""
        return {
            "openai_image_checker": OpenAIImageDetector,
            "q16_detector": Q16Detector,
            "multihead_detector": MultiheadDetector,
            "finetuned_q16_detector": FinetunedQ16Detector,
            "sd_safety_checker": SD_SafetyCheckerDetector,
            "azure_image_checker": AzureContentSafetyDetector,
            "llava_guard": LlavaGuardChecker,
            "gpt_4o": GPT_4o_ImageChecker
        }
        
    def __call__(self, prompt: str, attack_prompt: str = None, **kwargs) -> AttackResult:
        """
        Run attack with the specified prompt.
        
        Args:
            prompt: Input prompt to use for attack
            attack_prompt: Optional pre-crafted attack prompt
            **kwargs: Additional arguments passed to attack method
            
        Returns:
            AttackResult containing attack results and bypass status
        """
        return self.attacker.run_attack_with_checks(prompt, attack_prompt, **kwargs)
    
    @classmethod
    def register(cls, name: str, attacker_class: Type[BaseAttacker]) -> None:
        """
        Register a new attacker class.
        
        Args:
            name: Name to register the attacker under
            attacker_class: The attacker class to register
        """
        cls._registry[name] = attacker_class
    
    @classmethod
    def list_available_attackers(cls) -> List[str]:
        """Get list of all registered attacker names."""
        return list(cls._registry.keys())
    
    def attack_batch(self, prompts: List[str], **kwargs) -> List[AttackResult]:
        """Run attack on a batch of prompts"""
        results = self.attacker.attack_batch(prompts, **kwargs)
        return results
    
    def attack(self, prompt, **kwargs) -> AttackResult:
        """Run attack on a single prompt"""
        result = self.attacker.attack(prompt, **kwargs)
        return result