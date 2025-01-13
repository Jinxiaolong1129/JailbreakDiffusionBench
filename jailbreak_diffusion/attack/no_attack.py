from typing import Any, Optional, Callable, Dict, List
from .base import BaseAttacker, AttackResult
from .factory import AttackerFactory
import time



class NoAttacker(BaseAttacker):
    """
    A baseline attacker that performs no perturbation on input prompts.
    Useful for establishing baseline performance and control experiments.
    """
    
    def __init__(
        self,
        target_model: Any,
        text_detector: Optional[Callable[[str], bool]] = None,
        image_checker: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ) -> None:
        """
        Initialize the NoAttacker.
        
        Args:
            target_model: Model to generate images from
            text_detector: Optional function to detect harmful prompts
            image_checker: Optional function to detect harmful images
            **kwargs: Additional arguments (unused but accepted for compatibility)
        """
        super().__init__(
            target_model=target_model,
            text_detector=text_detector,
            image_checker=image_checker
        )
        
    def generate_perturbed_prompt(self, prompt: str, **kwargs) -> str:
        """
        Return the original prompt without any perturbation.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments (unused)
            
        Returns:
            The original prompt unchanged
        """
        return prompt
    
    def attack(self, prompt: str, **kwargs) -> AttackResult:
        """
        Main attack method that just passes through the prompt without modification.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters passed to generation
            
        Returns:
            AttackResult containing generation results
        """
        # Generate image with unmodified prompt
        start_time = time.time()
        
        generation_output = self.target_model.generate([prompt], **kwargs)
        
        end_time = time.time()

        return AttackResult(
            success=True,
            original_prompt=prompt,
            attack_prompt=prompt,
            execution_time=end_time - start_time,
            generated_image=generation_output.images,
            bypass_checker=True,
            bypass_detector=True,
            metadata={
                "attack_type": "no_attack",
                "model_metadata": generation_output.metadata
            }
        )
        
    def run_attack(self, prompt: str, **kwargs) -> AttackResult:
        """
        Run the "no attack" which just generates from the original prompt.
        Kept for compatibility with older code.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters passed to model
            
        Returns:
            AttackResult containing the generation results
        """
        return self.attack(prompt, **kwargs)

# Register the NoAttacker with the factory
AttackerFactory.register("no_attack", NoAttacker)