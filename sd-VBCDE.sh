#!/bin/bash

# "config/no_attack/harmful/VBCDE/stable-diffusion-3-medium.yaml"


MODELS=(
  # "config/no_attack/harmful/VBCDE/stable-diffusion-v1-5.yaml"
  # "config/no_attack/harmful/VBCDE/stable-diffusion-2.yaml"
  # "config/no_attack/harmful/VBCDE/stable-diffusion-xl-base-0.9.yaml"
  # "config/no_attack/harmful/VBCDE/stable-diffusion-3.5-large-turbo.yaml"
  # "config/no_attack/harmful/VBCDE/stable-diffusion-3.5-medium.yaml"
  # "config/no_attack/harmful/VBCDE/stable-diffusion-3.5-large.yaml"
  
  # "config/no_attack/harmful/VBCDE/hunyuan-dit-v1.2-distilled.yaml"

  "config/no_attack/harmful/VBCDE/flux-1-dev.yaml"
  # "config/no_attack/harmful/VBCDE/cogview.yaml"q
)

# Available GPUs as specified
AVAILABLE_GPUS=(1)

# Create logs directory if it doesn't exist
mkdir -p logs/VBCDE

# Run each model on a separate GPU from the available ones
for i in "${!MODELS[@]}"; do
  if [ $i -ge ${#AVAILABLE_GPUS[@]} ]; then
    echo "Warning: More models than available GPUs. Skipping ${MODELS[$i]}"
    continue
  fi
  
  MODEL=${MODELS[$i]}
  MODEL_NAME=$(basename "${MODEL}" .yaml)
  GPU_ID=${AVAILABLE_GPUS[$i]}
  
  echo "Starting experiment for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Set CUDA_VISIBLE_DEVICES to use only one specific GPU
  # Run the script with nohup to keep it running after terminal closes
  nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python exp_no_attack.py \
    --config_path "${MODEL}" > "logs/VBCDE/${MODEL_NAME}.log" 2>&1 &
  
  # Print the process ID for reference
  echo "Started process $! for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Wait a few seconds between launching processes
  sleep 2
done

echo "All experiments launched. Check logs in logs/VBCDE/ directory."

# Commented out example for manual launch if needed:
# CUDA_VISIBLE_DEVICES=7 nohup python exp_no_attack.py --config_path config/no_attack/harmful/VBCDE/flux-1-dev.yaml > logs/VBCDE/flux-1-dev.log 2>&1 &