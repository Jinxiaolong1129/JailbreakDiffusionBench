from google.cloud import language_v2
from typing import List, Dict

class GoogleTextModerator:
    """Google Cloud Natural Language API 的文本内容检测器类"""

    def __init__(self):
        """
        初始化检测器
        """
        self.client = language_v2.LanguageServiceClient()

    def moderate_text(self, text: str) -> Dict[str, float]:
        """
        检测单个文本内容

        Args:
            text: 需要检测的文本内容

        Returns:
            Dict[str, float]: 检测结果，包含各类别的置信度得分
        """
        document = {
            "content": text,
            "type_": language_v2.Document.Type.PLAIN_TEXT,
        }
        response = self.client.moderate_text(document=document)
        return {category.name: category.confidence for category in response.moderation_categories}

    def moderate_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        批量检测多个文本内容

        Args:
            texts: 需要检测的文本列表

        Returns:
            List[Dict[str, float]]: 检测结果列表，每个结果包含各类别的置信度得分
        """
        results = []
        for text in texts:
            results.append(self.moderate_text(text))
        return results

    def is_flagged(self, moderation_result: Dict[str, float], threshold: float = 0.5) -> bool:
        """
        判断内容是否被标记为违规

        Args:
            moderation_result: 检测结果，包含各类别的置信度得分
            threshold: 置信度阈值，超过此值则视为违规

        Returns:
            bool: 是否被标记为违规
        """
        return any(confidence >= threshold for confidence in moderation_result.values())

    def get_violations(self, moderation_result: Dict[str, float], threshold: float = 0.5) -> List[str]:
        """
        获取检测结果中违规的类别

        Args:
            moderation_result: 检测结果，包含各类别的置信度得分
            threshold: 置信度阈值，超过此值则视为违规

        Returns:
            List[str]: 违规类别列表
        """
        return [category for category, confidence in moderation_result.items() if confidence >= threshold]

# 示例用法
if __name__ == "__main__":
    moderator = GoogleTextModerator()
    text = "示例文本内容。"
    result = moderator.moderate_text(text)
    print("检测结果:", result)
    if moderator.is_flagged(result):
        print("检测到违规内容，类别:", moderator.get_violations(result))
    else:
        print("未检测到违规内容。")
