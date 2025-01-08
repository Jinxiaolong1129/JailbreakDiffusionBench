from typing import Any, Dict, List, Type
from .base import BaseAttacker, AttackResult

class AttackerFactory:
    _registry: Dict[str, Type[BaseAttacker]] = {}
    
    def __init__(self, attack_type: str, target_model: Any, **kwargs) -> None:
        if attack_type not in self._registry:
            raise ValueError(f"Attack type '{attack_type}' not found. Available: {list(self._registry.keys())}")
            
        attacker_class = self._registry[attack_type]
        self.attacker = attacker_class(target_model, **kwargs)
        
    def attack(self, prompt: str, **kwargs) -> AttackResult:
        return self.attacker.attack(prompt, **kwargs)
    
    @classmethod
    def register(cls, name: str, attacker_class: Type[BaseAttacker]) -> None:
        cls._registry[name] = attacker_class
    
    @classmethod
    def list_available_attackers(cls) -> List[str]:
        return list(cls._registry.keys())