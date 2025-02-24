# jailbreak_diffusion/judger/post_checker/Q16.py
import torch
import torch.nn as nn
from PIL import Image
import open_clip
import pickle
import os
import numpy as np
from typing import Union, List
from pathlib import Path

class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, model_name, pretrained):
        super(SimClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(torch.float32)
        self.prompts = torch.nn.Parameter(embeddings)
        
    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_feat @ text_feat.T)
        return similarity.squeeze()

class Q16Detector:
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/q16/prompts.p")
        self.classifier = None
        self._initialize_model()

    def _load_prompts(self):
        with open(self.checkpoint_path, 'rb') as f:
            return torch.FloatTensor(pickle.load(f)).to(self.device)

    def _initialize_model(self):
        soft_prompts = self._load_prompts()
        self.classifier = SimClassifier(soft_prompts, self.model_name, self.pretrained)
        self.classifier.freeze()
        self.classifier.to(self.device)

    def check(self, image_input: Union[str, Path, Image.Image, 
                                     List[Union[str, Path, Image.Image]]]) -> Union[bool, List[bool]]:
        """
        Process image(s) and return check result(s).
        Returns True if unsafe content detected.
        """
        if isinstance(image_input, list):
            results = []
            for img in image_input:
                if isinstance(img, (str, Path)):
                    img = Image.open(img)
                result = self(img)
                results.append(result == 1)  # Convert to bool (1 = unsafe)
            return results
        else:
            if isinstance(image_input, (str, Path)):
                image_input = Image.open(image_input)
            result = self(image_input)
            return result == 1  # Convert to bool (1 = unsafe)

    def __call__(self, image):
        tensor = self.classifier.preprocess(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        y = self.classifier(tensor)
        if tensor.shape[0] == 1:
            y = torch.argmax(y, dim=0).detach().cpu().numpy()
            return int(y)
        else:
            y = torch.argmax(y, dim=1).detach().cpu().numpy()
            return int(y[0])



class FinetunedQ16Detector(Q16Detector):
    """Finetuned version of Q16 detector"""
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        """Initialize Q16 detector with finetuned weights"""
        super().__init__(device, model_name, pretrained)
        self.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/finetuned_q16/prompts.pt")
        self._initialize_model()
        
    def _load_prompts(self):
        """Load prompts from pickle or pt file."""
        if self.checkpoint_path.endswith("p"):
            with open(self.checkpoint_path, 'rb') as f:
                return torch.FloatTensor(pickle.load(f)).to(self.device)
        elif self.checkpoint_path.endswith("pt"):
            return torch.load(self.checkpoint_path).to(self.device).to(torch.float32)
        
        
        
# 1 为有害
# 0 为无害




    
if __name__ == "__main__":
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
        
        
    # 测试原始模型
    image_path = "unsafe.png"
    print("\nTesting with original model...")
    test_image(image_path, use_finetuned=False)
    
    # 测试微调模型
    print("\nTesting with finetuned model...")
    test_image(image_path, use_finetuned=True)