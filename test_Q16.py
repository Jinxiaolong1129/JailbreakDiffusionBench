import torch
from PIL import Image
import PIL
import os
from jailbreak_diffusion.judger.post_detector.Q16 import Q16Detector, FinetunedQ16Detector


# 测试代码
def test_image(image_path, use_finetuned=False):
    """
    Test an image using Q16Detector
    Args:
        image_path: Path to the image file
        use_finetuned: Whether to use the finetuned model
    """
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 加载图片
        image = PIL.Image.open(image_path)
        # Image.open(image_path).convert('RGB')
        print(f"Loaded input image of size {image.size} from {image_path}")
        print(image)
        # 初始化检测器
        if use_finetuned:
            detector = FinetunedQ16Detector(device=device)
            model_type = "Finetuned Q16"
        else:
            detector = Q16Detector(device=device)
            model_type = "Original Q16"
            
        # 运行检测
        result = detector(image)
        
        # 打印结果
        print(f"\nResults from {model_type} Detector:")
        print(f"Image: {image_path}")
        print(f"Classification Result: {result}")
        
    except Exception as e:
        print(f"Error occurred during detection: {str(e)}")
        raise
    
    
if __name__ == "__main__":
    # 测试原始模型
    image_path = "unsafe.png"
    print("\nTesting with original model...")
    test_image(image_path, use_finetuned=False)
    
    # 测试微调模型
    print("\nTesting with finetuned model...")
    test_image(image_path, use_finetuned=True)