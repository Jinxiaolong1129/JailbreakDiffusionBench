#!/bin/bash

# List of Stable Diffusion models in the diffusion_db directory
MODELS=(
  # "config/no_attack/harmful/diffusion_db/hunyuan-dit-v1.2-distilled.yaml"
  # "config/no_attack/harmful/diffusion_db/stable-diffusion-2.yaml"
  # "config/no_attack/harmful/diffusion_db/stable-diffusion-xl-base-0.9.yaml"
  # "config/no_attack/harmful/diffusion_db/stable-diffusion-3.5-large-turbo.yaml"


  "config/no_attack/harmful/diffusion_db/flux-1-dev.yaml"
  "config/no_attack/harmful/diffusion_db/stable-diffusion-3-medium.yaml"
  "config/no_attack/harmful/diffusion_db/stable-diffusion-3.5-medium.yaml"
  "config/no_attack/harmful/diffusion_db/stable-diffusion-3.5-large.yaml"
)

# Available GPUs
AVAILABLE_GPUS=(2 4 5 7)

# Create logs directory if it doesn't exist
mkdir -p logs/harmful_diffusion_db

# Run each model on a separate GPU from the available ones
for i in "${!MODELS[@]}"; do
  if [ $i -ge ${#AVAILABLE_GPUS[@]} ]; then
    echo "Warning: Not enough GPUs available for all models!"
    break
  fi
  
  MODEL=${MODELS[$i]}
  MODEL_NAME=$(basename "${MODEL}" .yaml)
  GPU_ID=${AVAILABLE_GPUS[$i]}
  
  echo "Starting experiment for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Set CUDA_VISIBLE_DEVICES to use only one specific GPU
  # Run the script with nohup to keep it running after terminal closes
  nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python exp_no_attack.py \
    --config_path "${MODEL}" > "logs/harmful_diffusion_db/${MODEL_NAME}.log" 2>&1 &
  
  # Print the process ID for reference
  echo "Started process $! for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Wait a few seconds between launching processes
  sleep 2
done

echo "All experiments launched. Check logs in logs/harmful_diffusion_db/ directory."

# Individual commands for reference:
# CUDA_VISIBLE_DEVICES=0 nohup python exp_no_attack.py --config_path config/no_attack/harmful/diffusion_db/stable-diffusion-3.5-medium.yaml > logs/harmful_diffusion_db/stable-diffusion-3.5-medium.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python exp_no_attack.py --config_path config/no_attack/harmful/diffusion_db/stable-diffusion-3.5-large.yaml > logs/harmful_diffusion_db/stable-diffusion-3.5-large.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python exp_no_attack.py --config_path config/no_attack/harmful/diffusion_db/flux-1-dev.yaml > logs/harmful_diffusion_db/flux-1-dev.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python exp_no_attack.py --config_path config/no_attack/harmful/diffusion_db/hunyuan-dit-v1.2-distilled.yaml > logs/harmful_diffusion_db/hunyuan-dit-v1.2-distilled.log 2>&1 &


# CUDA_VISIBLE_DEVICES=7 nohup python exp_no_attack.py --config_path config/no_attack/harmful/diffusion_db/cogview.yaml > logs/harmful_diffusion_db/cogview.log 2>&1 &
