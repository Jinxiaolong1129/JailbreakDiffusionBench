from jailbreak_diffusion import DiffusionFactory
import os
from pathlib import Path
from jailbreak_diffusion.attack import AttackerFactory
import csv


def read_api_key(filename):
    """Read API key from file"""
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

def test_diffusion_models():
    # Read API keys
    dalle_api_key = read_api_key('api_dalle.txt')
    stability_api_key = read_api_key('api_stability.txt')
    
    base_save_dir = "test_outputs"
    test_prompt = ["A serene landscape with mountains and a lake at sunset"]
    
    Path(base_save_dir).mkdir(exist_ok=True)
    
    models_to_test = [
        # {
        #     "name": "stable-diffusion-3.5-medium",
        #     "params": {
        #         "model_name": "stable-diffusion-3.5-medium",
        #         "device": "cuda"  
        #     }
        # },
        # {
        #     "name": "stable-diffusion-xl-base-0.9",
        #     "params": {
        #     "model_name": "stable-diffusion-xl-base-0.9",
        #     "device": "cuda"
        #     }
        # },
        # {
        #     "name": "stable-diffusion-2",
        #     "params": {
        #         "model_name": "stable-diffusion-2",
        #         "device": "cuda"  
        #     }
        # },     
        # {
        #     "name": "flux-1-dev",
        #     "params": {
        #         "model_name": "flux-1-dev",
        #         "device": "cuda"  
        #     }
        # },
        {
            "name": "proteus-rundiffusion",
            "params": {
                "model_name": "proteus-rundiffusion",
                "device": "cuda"
            }
        },
        {
            "name": "cogview3",
            "params": {
                "model_name": "cogview3",
                "device": "cuda"  
            }
        },
    ]
    
    
    
    def read_csv(csv_file):
        """Read CSV file and return dictionary of index:prompt pairs"""
        prompts_dict = {}
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                for row in reader:
                    index, prompt, subclass = row  # Unpack the three columns
                    prompts_dict[index] = prompt
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found")
        return prompts_dict

    def generate_images(prompts_dict, model, save_dir, csv_writer):
        """Generate images using the model for each prompt"""
        for index, prompt in prompts_dict.items():
            output = model.generate([prompt])
            image_path = os.path.join(save_dir, f"{index}.png")
            output.images[0].save(image_path)
            csv_writer.writerow([index, prompt, image_path])
                
            print(f"Generation successful!")
            print(f"Generated image for prompt: {prompt}")
            print(f"Image saved as: {image_path}")
            print(f"Generation time: {output.metadata['generation_time']:.2f} seconds")
            print(f"Model parameters: {output.metadata['parameters']}")
            # break
        
    csv_file = "/home/ubuntu/xiaolong/jailbreakbench/data/I2P/04-Sex.csv"
    prompts_dict = read_csv(csv_file)
    
    for model_config in models_to_test:
        model_save_dir = os.path.join(base_save_dir, model_config["name"])
        Path(model_save_dir).mkdir(exist_ok=True)
        output_csv_file = os.path.join(model_save_dir, "generation_results.csv")

        with open(output_csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Index", "Prompt", "Image Save Dir"])
            
            print(f"\nTesting {model_config['name']}...")
            model = DiffusionFactory(**model_config["params"])
            attacker = AttackerFactory(attack_type="PGJ", target_model=model)
            
            generate_images(prompts_dict, model, model_save_dir, csv_writer)




if __name__ == "__main__":
    test_diffusion_models()