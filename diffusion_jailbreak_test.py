from jailbreak_diffusion import DiffusionFactory
import os
from pathlib import Path

def read_api_key(filename):
    """Read API key from file"""
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

def test_diffusion_models():
    """Test different types of diffusion models"""
    # Read API keys
    dalle_api_key = read_api_key('api_dalle.txt')
    stability_api_key = read_api_key('api_stability.txt')
    
    # Test configuration
    save_dir = "test_outputs"
    test_prompt = ["A serene landscape with mountains and a lake at sunset"]
    
    # Create output directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Test models
    models_to_test = [
        # Test local model
        {
            "name": "stable-diffusion-3-medium",
            "params": {
                "model_name": "stable-diffusion-3-medium",
                "save_dir": save_dir,
                "device": "cuda"  # Use "cpu" if no GPU available
            }
        },
        # # Test DALL-E model
        # {
        #     "name": "dalle-3",
        #     "params": {
        #         "model_name": "dalle-3",
        #         "api_key": dalle_api_key,
        #         "save_dir": save_dir
        #     }
        # },
        # # Test Stability AI model
        # {
        #     "name": "sdxl",
        #     "params": {
        #         "model_name": "sdxl",
        #         "api_key": stability_api_key,
        #         "save_dir": save_dir
        #     }
        # }
    ]
    
    for model_config in models_to_test:
        # try:
        print(f"\nTesting {model_config['name']}...")
        
        # Skip if API key is missing for API models
        if 'api_key' in model_config['params'] and not model_config['params']['api_key']:
            print(f"Skipping {model_config['name']} - API key not found")
            continue
        
        # Initialize model
        model = DiffusionFactory(**model_config["params"])
        
        # Generate image
        output = model.generate(test_prompt)
        
        # Print generation results
        print(f"Generation successful!")
        print(f"Number of images generated: {len(output.images)}")
        print(f"Images saved to: {save_dir}")
        print(f"Generation time: {output.metadata['generation_time']:.2f} seconds")
        print(f"Model parameters: {output.metadata['parameters']}")
        
        # except Exception as e:
        #     print(f"Error testing {model_config['name']}: {str(e)}")

if __name__ == "__main__":
    test_diffusion_models()