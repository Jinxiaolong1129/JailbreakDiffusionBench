from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ModerationResult:
    """存储审核结果的数据类"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    category_applied_input_types: Dict[str, List[str]]
    
class OpenAIImageDetector:
    """OpenAI图像内容检测器类"""
    
    def __init__(self):
        """
        初始化检测器。
        API密钥将从环境变量 OPENAI_API_KEY 中获取
        """
        self.client = OpenAI()
        self.model = "omni-moderation-latest"
        
    def detect_image(self, image_url: str, text: Optional[str] = None) -> ModerationResult:
        """
        检测图像内容是否包含不当内容
        
        Args:
            image_url: 图像URL或base64编码的图像数据
            text: 可选的配套文本内容
            
        Returns:
            ModerationResult: 包含检测结果的数据类实例
        """
        # 构建输入
        inputs = []
        if text:
            inputs.append({"type": "text", "text": text})
        
        inputs.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })
        
        # 调用API
        try:
            response = self.client.moderations.create(
                model=self.model,
                input=inputs
            )
            
            # 获取第一个结果
            result = response.results[0]
            
            return ModerationResult(
                flagged=result.flagged,
                categories=result.categories,
                category_scores=result.category_scores,
                category_applied_input_types=result.category_applied_input_types
            )
            
        except Exception as e:
            raise Exception(f"检测失败: {str(e)}")
    
    def is_safe_image(self, image_url: str, text: Optional[str] = None) -> bool:
        """
        快速检查图像是否安全
        
        Args:
            image_url: 图像URL或base64编码的图像数据
            text: 可选的配套文本内容
            
        Returns:
            bool: 如果图像安全返回True，否则返回False
        """
        result = self.detect_image(image_url, text)
        return not result.flagged
    
    def get_violation_details(self, result: ModerationResult) -> Dict[str, float]:
        """
        获取违规详情
        
        Args:
            result: 检测结果
            
        Returns:
            Dict[str, float]: 违规类别及其置信度分数的字典
        """
        return {
            category: score
            for category, score in result.category_scores.items()
            if result.categories[category]
        }

# 使用示例
if __name__ == "__main__":
    detector = OpenAIImageDetector()
    
    # 检测单个图片
    result = detector.detect_image(
        "https://example.com/image.png",
        "检查这张图片"
    )
    
    # 检查是否安全
    is_safe = detector.is_safe_image("https://example.com/image.png")
    
    # 获取违规详情
    if not is_safe:
        violations = detector.get_violation_details(result)
        print("检测到的违规内容:", violations)