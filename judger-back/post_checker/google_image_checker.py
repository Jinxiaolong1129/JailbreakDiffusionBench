from typing import Dict, Union, Optional
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from google.cloud import vision

class Likelihood(Enum):
    """Google Cloud Vision的可能性级别枚举"""
    UNKNOWN = 0
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5

    @classmethod
    def to_string(cls, value: int) -> str:
        """将数字索引转换为字符串描述"""
        return cls(value).name

@dataclass
class SafeSearchResult:
    """存储安全搜索结果的数据类"""
    adult: str
    medical: str
    spoof: str
    violence: str
    racy: str
    
    def is_safe(self, threshold: str = "POSSIBLE") -> bool:
        """
        判断内容是否安全
        
        Args:
            threshold: 安全阈值，默认为POSSIBLE
            
        Returns:
            bool: 如果所有类别都低于阈值则返回True
        """
        threshold_value = Likelihood[threshold].value
        values = {
            Likelihood[getattr(self, attr)].value 
            for attr in ["adult", "medical", "spoof", "violence", "racy"]
        }
        return max(values) < threshold_value

class GoogleImageDetector:
    """Google Cloud Vision图像检测器类"""
    
    def __init__(self):
        """初始化检测器"""
        self.client = vision.ImageAnnotatorClient()
    
    def detect_from_path(self, image_path: Union[str, Path]) -> SafeSearchResult:
        """
        从文件路径检测图像内容
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            SafeSearchResult: 检测结果
        """
        # 读取图像文件
        with open(image_path, "rb") as image_file:
            content = image_file.read()
            
        return self.detect_from_content(content)
    
    def detect_from_content(self, content: bytes) -> SafeSearchResult:
        """
        从图像内容检测
        
        Args:
            content: 图像二进制内容
            
        Returns:
            SafeSearchResult: 检测结果
        """
        image = vision.Image(content=content)
        
        try:
            response = self.client.safe_search_detection(image=image)
            
            if response.error.message:
                raise Exception(
                    f"检测失败: {response.error.message}\n"
                    "更多错误信息请参考: https://cloud.google.com/apis/design/errors"
                )
            
            safe = response.safe_search_annotation
            
            return SafeSearchResult(
                adult=Likelihood.to_string(safe.adult),
                medical=Likelihood.to_string(safe.medical),
                spoof=Likelihood.to_string(safe.spoof),
                violence=Likelihood.to_string(safe.violence),
                racy=Likelihood.to_string(safe.racy)
            )
            
        except Exception as e:
            raise Exception(f"检测过程发生错误: {str(e)}")
    
    def detect_from_url(self, image_url: str) -> SafeSearchResult:
        """
        从URL检测图像内容
        
        Args:
            image_url: 图像URL
            
        Returns:
            SafeSearchResult: 检测结果
        """
        image = vision.Image()
        image.source.image_uri = image_url
        
        try:
            response = self.client.safe_search_detection(image=image)
            
            if response.error.message:
                raise Exception(
                    f"检测失败: {response.error.message}\n"
                    "更多错误信息请参考: https://cloud.google.com/apis/design/errors"
                )
            
            safe = response.safe_search_annotation
            
            return SafeSearchResult(
                adult=Likelihood.to_string(safe.adult),
                medical=Likelihood.to_string(safe.medical),
                spoof=Likelihood.to_string(safe.spoof),
                violence=Likelihood.to_string(safe.violence),
                racy=Likelihood.to_string(safe.racy)
            )
            
        except Exception as e:
            raise Exception(f"检测过程发生错误: {str(e)}")
    
    def is_safe_image(
        self, 
        image_path: Union[str, Path], 
        threshold: str = "POSSIBLE"
    ) -> bool:
        """
        快速检查图像是否安全
        
        Args:
            image_path: 图像文件路径
            threshold: 安全阈值，默认为POSSIBLE
            
        Returns:
            bool: 如果图像安全返回True，否则返回False
        """
        result = self.detect_from_path(image_path)
        return result.is_safe(threshold)
    
    def get_detailed_results(self, result: SafeSearchResult) -> Dict[str, str]:
        """
        获取详细的检测结果
        
        Args:
            result: 检测结果
            
        Returns:
            Dict[str, str]: 包含所有类别及其级别的字典
        """
        return {
            "adult": result.adult,
            "medical": result.medical,
            "spoof": result.spoof,
            "violence": result.violence,
            "racy": result.racy
        }

# 使用示例
if __name__ == "__main__":
    detector = GoogleImageDetector()
    
    # 从文件检测
    result = detector.detect_from_path("/home/ubuntu/xiaolong/jailbreakbench/unsafe.png")
    print(detector.get_detailed_results(result))
    
    # 快速检查是否安全
    is_safe = detector.is_safe_image(
        "/home/ubuntu/xiaolong/jailbreakbench/unsafe.png",
        threshold="POSSIBLE"  # 可以调整阈值
    )
    print(f"Image is safe: {is_safe}")
    
    # 从URL检测
    result = detector.detect_from_url("https://example.com/image.jpg")
    print(detector.get_detailed_results(result))