import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import open_clip
import os

class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(device)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
        ).to(device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, image):
        x = self.clip_model.encode_image(image).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out

class MultiheadDetector:
    def __init__(self, device, model_name="ViT-L-14", pretrained='openai'):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints/multi-headed")
        self.model = None
        self.unsafe_contents = ["sexual", "violent", "disturbing", "hateful", "political"]
        self._initialize_model()

    def _initialize_model(self):
        self.model = MHSafetyClassifier(self.device, self.model_name, self.pretrained)
        self.model.freeze()

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
                results.append(result['unsafe'])
            return results
        else:
            if isinstance(image_input, (str, Path)):
                image_input = Image.open(image_input)
            result = self(image_input)
            return result['unsafe']

    def __call__(self, image):
        tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
        return self._process_single(tensor[0].unsqueeze(0))

    def _process_single(self, tensor):
        results = {}
        with torch.no_grad():
            head_results = []
            for head in self.unsafe_contents:
                self._load_head(head)
                logits = self.model(tensor).squeeze()
                pred = (logits.detach().cpu() > 0.5).to(dtype=torch.int64)
                results[head] = bool(pred.item())
                head_results.append(pred.item())
                
        results['unsafe'] = bool(sum(head_results) > 0)
        return results

    def _load_head(self, head):
        checkpoint_path = os.path.join(self.checkpoints_dir, f"{head}.pt")
        self.model.projection_head.load_state_dict(torch.load(checkpoint_path))
        self.model.projection_head.eval()
        
        
        
        


if __name__ == "__main__":
    def test_image(image_path, model_name="ViT-L-14", pretrained="openai"):
        """
        Test an image using MultiheadDetector
        Args:
            image_path: Path to the image file
            model_name: Name of the CLIP model to use
            pretrained: Name of the pretrained weights
        """
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} does not exist!")
            return
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"Loaded input image of size {image.size} from {image_path}")
            
            # Initialize detector
            detector = MultiheadDetector(
                device=device,
                model_name=model_name,
                pretrained=pretrained
            )
            
            # Run detection
            result = detector(image)
            
            # Print results
            print("\nMultihead Detection Results:")
            print(f"Image: {image_path}")
            print("\nDetection Results by Category:")
            for category, is_detected in result.items():
                if category != 'unsafe':
                    print(f"{category.capitalize()}: {'Detected' if is_detected else 'Not Detected'}")
            
            print(f"\nOverall Safety: {'Unsafe' if result['unsafe'] else 'Safe'}")
            
        except Exception as e:
            print(f"Error occurred during detection: {str(e)}")
            raise

    # Test single image
    print("\nTesting single image...")
    test_image("unsafe.png")
    