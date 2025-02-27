#!/bin/bash

# List of Stable Diffusion models in the harmful/diffusion_db directory
MODELS=(
  "config/no_attack/harmful/diffusion_db/hunyuan-dit-v1.2-distilled.yaml"
)

# Available GPU
AVAILABLE_GPU=1

# Create logs directory if it doesn't exist
mkdir -p logs/harmful_diffusion_db

# Run the model on GPU 1
MODEL=${MODELS[0]}
MODEL_NAME=$(basename "${MODEL}" .yaml)

echo "Starting experiment for ${MODEL_NAME} on GPU ${AVAILABLE_GPU}"

# Set CUDA_VISIBLE_DEVICES to use only one specific GPU
# Run the script with nohup to keep it running after terminal closes
nohup env CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU} python exp_no_attack.py \
  --config_path "${MODEL}" > "logs/harmful_diffusion_db/${MODEL_NAME}.log" 2>&1 &

# Print the process ID for reference
echo "Started process $! for ${MODEL_NAME} on GPU ${AVAILABLE_GPU}"

echo "Experiment launched. Check logs in logs/harmful_diffusion_db/ directory."

# For reference:
# CUDA_VISIBLE_DEVICES=1 nohup python exp_no_attack.py --config_path config/no_attack/harmful/diffusion_db/hunyuan-dit-v1.2-distilled.yaml > logs/harmful_diffusion_db/hunyuan-dit-v1.2-distilled.log 2>&1 &