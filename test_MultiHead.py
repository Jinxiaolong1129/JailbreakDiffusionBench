import torch
from PIL import Image
import PIL
import os
from jailbreak_diffusion.judger.post_checker.MultiheadDetector import MultiheadDetector

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


if __name__ == "__main__":
    # Test single image
    print("\nTesting single image...")
    test_image("unsafe.png")
    