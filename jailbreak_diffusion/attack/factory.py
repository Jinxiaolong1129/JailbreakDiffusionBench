from typing import Any, Dict, List, Type, Optional, Callable
from .base import BaseAttacker, AttackResult

class AttackerFactory:
    _registry: Dict[str, Type[BaseAttacker]] = {}
    
    def __init__(
        self, 
        attack_type: str, 
        target_model: Any,
        text_detector: Optional[Callable[[str], bool]] = None,
        image_checker: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ) -> None:
        """
        Initialize an attacker with specified type and checkers.
        
        Args:
            attack_type: Name of the attack method to use
            target_model: Model to be attacked
            text_detector: Function to detect harmful prompts
            image_checker: Function to detect harmful generated images
            **kwargs: Additional arguments passed to attacker
        """
        if attack_type not in self._registry:
            raise ValueError(f"Attack type '{attack_type}' not found. Available: {list(self._registry.keys())}")
            
        attacker_class = self._registry[attack_type]
        self.attacker = attacker_class(
            target_model=target_model,
            text_detector=text_detector,
            image_checker=image_checker,
            **kwargs
        )
        
    def attack(self, prompt: str, **kwargs) -> AttackResult:
        """
        Run attack with the specified prompt.
        
        Args:
            prompt: Input prompt to use for attack
            **kwargs: Additional arguments passed to attack method
            
        Returns:
            AttackResult containing attack results and bypass status
        """
        return self.attacker.run_attack_with_checks(prompt, **kwargs)
    
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