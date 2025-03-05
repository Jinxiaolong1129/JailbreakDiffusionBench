#!/bin/bash

# Script to run all RingABell jailbreak configs on different GPUs
# Usage: bash run_all_models.sh

# Create log directory if it doesn't exist
mkdir -p log/jailbreak/RingABell

# List of available GPUs
GPUS=(0 4 5)
# GPUS=(0 1 2 3 4 5 6 7)

# List of config files
CONFIG_FILES=(
  # "config/jailbreak/RingABell/cogview.yaml"
  # "config/jailbreak/RingABell/flux-1-dev.yaml"
  # "config/jailbreak/RingABell/hunyuan-dit-v1.2-distilled.yaml"
  # "config/jailbreak/RingABell/stable-diffusion-3.5-medium.yaml"
  # "config/jailbreak/RingABell/stable-diffusion-3-medium.yaml"

  "config/jailbreak/RingABell/stable-diffusion-3.5-large-turbo.yaml"
  "config/jailbreak/RingABell/stable-diffusion-v1-5.yaml"
  "config/jailbreak/RingABell/stable-diffusion-xl-base-0.9.yaml"
)

# Check if we have enough GPUs
if [ ${#CONFIG_FILES[@]} -gt ${#GPUS[@]} ]; then
  echo "Warning: More config files (${#CONFIG_FILES[@]}) than available GPUs (${#GPUS[@]})"
  echo "Some configurations will need to wait for others to complete"
fi

# Function to extract model name from config file
get_model_name() {
  local config_file=$1
  # Extract the filename without path and extension
  local basename=$(basename "$config_file" .yaml)
  echo "$basename"
}

# Launch each config on a different GPU
for i in "${!CONFIG_FILES[@]}"; do
  config=${CONFIG_FILES[$i]}
  gpu_idx=$((i % ${#GPUS[@]}))
  gpu=${GPUS[$gpu_idx]}
  
  model_name=$(get_model_name "$config")
  log_file="log/jailbreak/RingABell/${model_name}.log"
  
  echo "Starting $model_name on GPU $gpu, logging to $log_file"
  
  # Run with nohup, export CUDA_VISIBLE_DEVICES to select GPU
  nohup bash -c "export CUDA_VISIBLE_DEVICES=$gpu && python jailbreak_RingABell.py --config_path $config" > "$log_file" 2>&1 &
  
  # Add a small delay to avoid potential race conditions
  sleep 2
done

echo "All jobs submitted. Check the logs in log/jailbreak/RingABell/ directory"
echo "You can monitor progress with: tail -f log/jailbreak/RingABell/*.log"