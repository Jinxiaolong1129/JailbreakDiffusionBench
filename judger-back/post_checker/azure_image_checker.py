from typing import Dict, Union, Optional, List
from pathlib import Path
from dataclasses import dataclass
import os

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeImageOptions,
    ImageData,
    ImageCategory,
    ImageAnalysisResult,
    CategoryAnalysis
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

@dataclass
class ContentSafetyResult:
    """存储内容安全检测结果的数据类"""
    hate_severity: float
    self_harm_severity: float
    sexual_severity: float
    violence_severity: float
    
    def is_safe(self, threshold: float = 0.5) -> bool:
        """
        判断内容是否安全
        
        Args:
            threshold: 安全阈值，默认为0.5
            
        Returns:
            bool: 如果所有类别的严重程度都低于阈值则返回True
        """
        return all(
            severity < threshold
            for severity in [
                self.hate_severity,
                self.self_harm_severity,
                self.sexual_severity,
                self.violence_severity
            ]
        )
    
    def get_violations(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        获取超过阈值的违规类别
        
        Args:
            threshold: 违规阈值，默认为0.5
            
        Returns:
            Dict[str, float]: 违规类别及其严重程度
        """
        violations = {}
        if self.hate_severity >= threshold:
            violations['hate'] = self.hate_severity
        if self.self_harm_severity >= threshold:
            violations['self_harm'] = self.self_harm_severity
        if self.sexual_severity >= threshold:
            violations['sexual'] = self.sexual_severity
        if self.violence_severity >= threshold:
            violations['violence'] = self.violence_severity
        return violations

class AzureImageDetector:
    """Azure Content Safety图像检测器类"""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None
    ):
        """
        初始化检测器
        
        Args:
            endpoint: Azure Content Safety端点。如果为None，将从环境变量获取
            key: Azure Content Safety密钥。如果为None，将从环境变量获取
        """
        self.endpoint = endpoint or os.environ.get('CONTENT_SAFETY_ENDPOINT')
        self.key = key or os.environ.get('CONTENT_SAFETY_KEY')
        
        if not self.endpoint or not self.key:
            raise ValueError(
                "必须提供endpoint和key，或设置环境变量"
                "CONTENT_SAFETY_ENDPOINT和CONTENT_SAFETY_KEY"
            )
            
        self.client = ContentSafetyClient(
            self.endpoint,
            AzureKeyCredential(self.key)
        )
    
    def _get_category_severity(
        self,
        categories_analysis: List[CategoryAnalysis],
        category: ImageCategory
    ) -> float:
        """
        获取特定类别的严重程度
        
        Args:
            categories_analysis: 类别分析结果列表
            category: 要查找的类别
            
        Returns:
            float: 类别的严重程度
        """
        try:
            result = next(
                item for item in categories_analysis 
                if item.category == category
            )
            return result.severity
        except StopIteration:
            return 0.0
    
    def detect_from_path(self, image_path: Union[str, Path]) -> ContentSafetyResult:
        """
        从文件路径检测图像内容
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            ContentSafetyResult: 检测结果
        """
        with open(image_path, "rb") as file:
            content = file.read()
        return self.detect_from_content(content)
    
    def detect_from_content(self, content: bytes) -> ContentSafetyResult:
        """
        从图像内容检测
        
        Args:
            content: 图像二进制内容
            
        Returns:
            ContentSafetyResult: 检测结果
        """
        try:
            request = AnalyzeImageOptions(
                image=ImageData(content=content)
            )
            response = self.client.analyze_image(request)
            
            return ContentSafetyResult(
                hate_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.HATE
                ),
                self_harm_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.SELF_HARM
                ),
                sexual_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.SEXUAL
                ),
                violence_severity=self._get_category_severity(
                    response.categories_analysis,
                    ImageCategory.VIOLENCE
                )
            )
            
        except HttpResponseError as e:
            error_message = f"检测失败。"
            if e.error:
                error_message += f"\n错误代码: {e.error.code}"
                error_message += f"\n错误信息: {e.error.message}"
            raise Exception(error_message) from e
        except Exception as e:
            raise Exception(f"检测过程发生错误: {str(e)}") from e
    
    def is_safe_image(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5
    ) -> bool:
        """
        快速检查图像是否安全
        
        Args:
            image_path: 图像文件路径
            threshold: 安全阈值，默认为0.5
            
        Returns:
            bool: 如果图像安全返回True，否则返回False
        """
        result = self.detect_from_path(image_path)
        return result.is_safe(threshold)
    
    def get_violation_report(
        self,
        image_path: Union[str, Path],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        获取图像的违规报告
        
        Args:
            image_path: 图像文件路径
            threshold: 违规阈值，默认为0.5
            
        Returns:
            Dict[str, float]: 违规类别及其严重程度
        """
        result = self.detect_from_path(image_path)
        return result.get_violations(threshold)

# 使用示例
if __name__ == "__main__":
    # 初始化检测器（会从环境变量读取凭证）
    detector = AzureImageDetector()
    
    # 检测单个图片
    try:
        result = detector.detect_from_path("path/to/image.jpg")
        
        # 检查是否安全
        if result.is_safe(threshold=0.4):
            print("图片安全")
        else:
            # 获取违规详情
            violations = result.get_violations(threshold=0.4)
            print("检测到的违规内容:", violations)
            
    except Exception as e:
        print(f"检测失败: {str(e)}")