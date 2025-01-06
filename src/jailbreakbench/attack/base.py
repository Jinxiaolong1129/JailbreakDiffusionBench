from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class AttackResult:
    success: bool
    execution_time: float
    generated_image: Any
    metadata: Dict[str, Any]

class BaseAttacker(ABC):
    def __init__(self, target_model: Any, **kwargs):
        self.target_model = target_model
        self.name = self.__class__.__name__
        
    @abstractmethod
    def attack(self, prompt: str, **kwargs) -> AttackResult:
        pass