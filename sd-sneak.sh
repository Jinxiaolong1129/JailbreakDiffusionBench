#!/bin/bash

# "config/no_attack/harmful/sneakprompt/stable-diffusion-3-medium.yaml"
# "config/no_attack/harmful/sneakprompt/stable-diffusion-3.5-medium.yaml"
# "config/no_attack/harmful/sneakprompt/stable-diffusion-3.5-large.yaml"

# List of Stable Diffusion models to run
MODELS=(
  "config/no_attack/harmful/sneakprompt/stable-diffusion-v1-5.yaml"
  "config/no_attack/harmful/sneakprompt/stable-diffusion-2.yaml"
  "config/no_attack/harmful/sneakprompt/stable-diffusion-xl-base-0.9.yaml"
  "config/no_attack/harmful/sneakprompt/stable-diffusion-3.5-large-turbo.yaml"
)

# Available GPUs as specified
AVAILABLE_GPUS=(2 4 5 7)

# Create logs directory if it doesn't exist
mkdir -p logs/sneakprompt

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
    --config_path "${MODEL}" > "logs/sneakprompt/${MODEL_NAME}.log" 2>&1 &
  
  # Print the process ID for reference
  echo "Started process $! for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Wait a few seconds between launching processes
  sleep 2
done

echo "All experiments launched. Check logs in logs/sneakprompt/ directory."
