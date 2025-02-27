#!/bin/bash

# List of Stable Diffusion models in the benign/diffusion_db directory
MODELS=(
  "config/no_attack/benign/diffusion_db/stable-diffusion-v1-5.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-2.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-xl-base-0.9.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-safe.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-3-medium.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-3.5-medium.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-3.5-large.yaml"
  "config/no_attack/benign/diffusion_db/stable-diffusion-3.5-large-turbo.yaml"
)

# Create logs directory if it doesn't exist
mkdir -p logs/benign_diffusion_db

# Run each model on a separate GPU
for i in "${!MODELS[@]}"; do
  MODEL=${MODELS[$i]}
  MODEL_NAME=$(basename "${MODEL}" .yaml)
  GPU_ID=$i  # This will assign GPUs 0-7
  
  echo "Starting experiment for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Set CUDA_VISIBLE_DEVICES to use only one specific GPU
  # Run the script with nohup to keep it running after terminal closes
  nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python exp_no_attack.py \
    --config_path "${MODEL}" > "logs/benign_diffusion_db/${MODEL_NAME}.log" 2>&1 &
  
  # Print the process ID for reference
  echo "Started process $! for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Wait a few seconds between launching processes
  sleep 2
done

echo "All experiments launched. Check logs in logs/benign_diffusion_db/ directory."


# CUDA_VISIBLE_DEVICES=3 nohup python exp_no_attack.py --config_path config/no_attack/benign/diffusion_db/flux-1-dev.yaml > logs/benign_diffusion_db/flux-1-dev.log 2>&1 &