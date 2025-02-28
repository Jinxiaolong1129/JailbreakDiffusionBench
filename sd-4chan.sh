#!/bin/bash

# List of Stable Diffusion models in the 4chan directory
MODELS=(
  "config/no_attack/harmful/4chan/flux-1-dev.yaml"
  "config/no_attack/harmful/4chan/hunyuan-dit-v1.2-distilled.yaml"
)

# Available GPUs
AVAILABLE_GPUS=(1 2)

# Create logs directory if it doesn't exist
mkdir -p logs/4chan

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
    --config_path "${MODEL}" > "logs/4chan/${MODEL_NAME}.log" 2>&1 &
  
  # Print the process ID for reference
  echo "Started process $! for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Wait a few seconds between launching processes
  sleep 2
done

echo "All experiments launched. Check logs in logs/4chan/ directory."


# CUDA_VISIBLE_DEVICES=7 nohup python exp_no_attack.py --config_path config/no_attack/harmful/4chan/stable-diffusion-3.5-large-turbo.yaml > logs/4chan/stable-diffusion-3.5-large-turbo.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python exp_no_attack.py --config_path config/no_attack/harmful/4chan/flux-1-dev.yaml > logs/4chan/stable-diffusion-3.5-large-turbo.log 2>&1 &
