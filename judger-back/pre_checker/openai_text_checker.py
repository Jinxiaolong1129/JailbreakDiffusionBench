from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from openai import OpenAI
import json

@dataclass
class ModerationResult:
    """存储内容审核结果的数据类"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Optional[Dict[str, List[str]]] = None

class TextDetector:
    """OpenAI文本内容检测器类"""
    
    def __init__(self, api_key: str = None, model: str = "omni-moderation-latest"):
        """
        初始化检测器
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型,默认为最新的omni-moderation模型
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # 所有可能的检测类别
        self.categories = [
            "harassment", "harassment/threatening",
            "hate", "hate/threatening",
            "illicit", "illicit/violent",
            "self-harm", "self-harm/intent", "self-harm/instructions",
            "sexual", "sexual/minors",
            "violence", "violence/graphic"
        ]

    def detect(self, text: str) -> ModerationResult:
        """
        检测单个文本内容
        
        Args:
            text: 需要检测的文本内容
            
        Returns:
            ModerationResult: 检测结果
        """
        try:
            response = self.client.moderations.create(
                model=self.model,
                input=text
            )
            
            # 获取第一个结果(因为只传入了一个文本)
            result = response.results[0]
            
            return ModerationResult(
                flagged=result.flagged,
                categories=result.categories,
                category_scores=result.category_scores,
                category_applied_input_types=getattr(result, 'category_applied_input_types', None)
            )
            
        except Exception as e:
            raise Exception(f"文本检测失败: {str(e)}")

    def detect_batch(self, texts: List[str]) -> List[ModerationResult]:
        """
        批量检测多个文本内容
        
        Args:
            texts: 需要检测的文本列表
            
        Returns:
            List[ModerationResult]: 检测结果列表
        """
        try:
            response = self.client.moderations.create(
                model=self.model,
                input=texts
            )
            
            results = []
            for result in response.results:
                results.append(ModerationResult(
                    flagged=result.flagged,
                    categories=result.categories,
                    category_scores=result.category_scores,
                    category_applied_input_types=getattr(result, 'category_applied_input_types', None)
                ))
                
            return results
            
        except Exception as e:
            raise Exception(f"批量文本检测失败: {str(e)}")
            
    def get_violations(self, result: ModerationResult) -> List[str]:
        """
        获取检测结果中违规的类别
        
        Args:
            result: 检测结果
            
        Returns:
            List[str]: 违规类别列表
        """
        violations = []
        for category in self.categories:
            if result.categories.get(category, False):
                violations.append(category)
        return violations

    def is_safe(self, result: ModerationResult) -> bool:
        """
        判断内容是否安全(没有任何违规)
        
        Args:
            result: 检测结果
            
        Returns:
            bool: 是否安全
        """
        return not result.flagged