# base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
import time

@dataclass
class AttackResult:
    success: bool
    original_prompt: str
    attack_prompt: str
    execution_time: float
    generated_image: Any
    bypass_detector: bool  # Whether it bypassed the text detector
    bypass_checker: bool   # Whether it bypassed the image checker
    metadata: Dict[str, Any]

class BaseAttacker(ABC):
    def __init__(
        self, 
        target_model: Any,
        text_detector: Optional[Callable[[str], bool]] = None,
        image_checker: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ):
        self.target_model = target_model
        self.text_detector = text_detector
        self.image_checker = image_checker
        self.name = self.__class__.__name__
        
    def check_text(self, prompt: str) -> bool:
        """Check if prompt passes the text detector"""
        if self.text_detector is None:
            return True
        return self.text_detector(prompt)
    
    def check_image(self, image: Any) -> bool:
        """Check if generated image passes the image checker"""
        if self.image_checker is None:
            return True
        return self.image_checker(image)
        
    @abstractmethod
    def attack(self, prompt: str, **kwargs) -> AttackResult:
        pass
    
    def run_attack_with_checks(self, prompt: str, **kwargs) -> AttackResult:
        
        # TODO: 分类。方法需要 detector的反馈。和不需要detector的反馈
        """Template method to run attack with all checks"""
        start_time = time.time()
        
        # Check text detector first
        bypass_detector = self.check_text(prompt)
        
        # Run the actual attack implementation
        result = self.attack(prompt, **kwargs)
        
        # Check generated image
        bypass_checker = self.check_image(result.generated_image)
        
        execution_time = time.time() - start_time
        
        return AttackResult(
            success=result.success,
            original_prompt=prompt,
            attack_prompt=result.attack_prompt,
            execution_time=execution_time,
            generated_image=result.generated_image,
            bypass_detector=bypass_detector,
            bypass_checker=bypass_checker,
            metadata={
                **result.metadata,
                "bypass_checks": {
                    "text_detector": bypass_detector,
                    "image_checker": bypass_checker
                }
            }
        )