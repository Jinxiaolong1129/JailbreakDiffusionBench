from dataclasses import dataclass
from typing import Dict, List, Optional
import os
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

@dataclass
class CategoryResult:
    """存储每个类别的检测结果"""
    category: str
    severity: float

@dataclass
class DetectionResult:
    """存储完整的检测结果"""
    categories_analysis: List[CategoryResult]
    is_flagged: bool
    error: Optional[str] = None

class AzureTextDetector:
    """Azure Content Safety文本检测器类"""
    
    def __init__(self, key: str = None, endpoint: str = None):
        """
        初始化检测器
        
        Args:
            key: Azure Content Safety API密钥
            endpoint: API端点
        """
        self.key = key or os.environ.get("CONTENT_SAFETY_KEY")
        self.endpoint = endpoint or os.environ.get("CONTENT_SAFETY_ENDPOINT")
        
        if not self.key or not self.endpoint:
            raise ValueError("必须提供API密钥和端点，可以通过参数传入或设置环境变量")
            
        self.client = ContentSafetyClient(self.endpoint, AzureKeyCredential(self.key))
        
        # 定义检测类别
        self.categories = [
            TextCategory.HATE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.VIOLENCE
        ]
        
        # 严重程度阈值（可根据需求调整）
        self.severity_threshold = 2

    def detect(self, text: str) -> DetectionResult:
        """
        检测单个文本内容
        
        Args:
            text: 需要检测的文本
            
        Returns:
            DetectionResult: 检测结果
        """
        try:
            # 构造请求
            request = AnalyzeTextOptions(text=text)
            
            # 发送请求
            response = self.client.analyze_text(request)
            
            # 处理结果
            category_results = []
            is_flagged = False
            
            for category in self.categories:
                result = next(item for item in response.categories_analysis 
                            if item.category == category)
                
                if result:
                    # 如果严重程度超过阈值，标记为不安全
                    if result.severity >= self.severity_threshold:
                        is_flagged = True
                        
                    category_results.append(CategoryResult(
                        category=result.category.name,
                        severity=result.severity
                    ))
            
            return DetectionResult(
                categories_analysis=category_results,
                is_flagged=is_flagged
            )
            
        except HttpResponseError as e:
            error_message = f"错误代码: {e.error.code}, 错误信息: {e.error.message}" if e.error else str(e)
            return DetectionResult(
                categories_analysis=[],
                is_flagged=False,
                error=error_message
            )
        except Exception as e:
            return DetectionResult(
                categories_analysis=[],
                is_flagged=False,
                error=str(e)
            )

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """
        批量检测多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        return [self.detect(text) for text in texts]

    def get_highest_severity(self, result: DetectionResult) -> Optional[CategoryResult]:
        """
        获取检测结果中严重程度最高的类别
        
        Args:
            result: 检测结果
            
        Returns:
            Optional[CategoryResult]: 严重程度最高的类别结果
        """
        if not result.categories_analysis:
            return None
            
        return max(result.categories_analysis, key=lambda x: x.severity)

    def is_safe(self, result: DetectionResult) -> bool:
        """
        判断内容是否安全
        
        Args:
            result: 检测结果
            
        Returns:
            bool: 是否安全
        """
        return not result.is_flagged and not result.error

    def get_unsafe_categories(self, result: DetectionResult) -> List[str]:
        """
        获取超过安全阈值的类别
        
        Args:
            result: 检测结果
            
        Returns:
            List[str]: 不安全类别列表
        """
        return [cat.category for cat in result.categories_analysis 
                if cat.severity >= self.severity_threshold]