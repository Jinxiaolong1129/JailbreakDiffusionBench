import yaml
import os

config_dir = 'config'
yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]

for yaml_file in yaml_files:
    yaml_path = os.path.join(config_dir, yaml_file)
        
    print(f"Loading {yaml_path}")
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # Convert model dict to ModelConfig
    model_dict = config_dict.pop('model')
    print(model_dict)    
    print(f"Successfully loaded {yaml_file}")
