#!/bin/bash

# Configuration
TOTAL_IDS=400
CONFIG_PATH="config/MMA/stable-diffusion-3.5-medium.yaml"

# Create a directory for logs if it doesn't exist
mkdir -p logs/MMA/

# Explicitly define GPU IDs
GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_IDS[@]}
IDS_PER_GPU=$((TOTAL_IDS / GPU_COUNT))

echo "=============== Commands to be executed ==============="
# First print all the commands
for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    
    # Calculate start and end IDs for this GPU
    start_id=$((i * IDS_PER_GPU + 1))
    end_id=$(((i + 1) * IDS_PER_GPU))
    
    # Create the command - environment variable needs to be set differently for nohup
    CMD="python exp_MMA.py --config_path $CONFIG_PATH --start_id $start_id --end_id $end_id"
    
    echo "GPU $gpu_id: CUDA_VISIBLE_DEVICES=$gpu_id $CMD"
done
echo "======================================================="

echo -e "\nStarting execution...\n"

# Then execute them
for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    
    # Calculate start and end IDs for this GPU
    start_id=$((i * IDS_PER_GPU + 1))
    end_id=$(((i + 1) * IDS_PER_GPU))
    
    # Run the command with nohup and save logs
    echo "Starting job on GPU $gpu_id: Processing IDs $start_id to $end_id"
    nohup env CUDA_VISIBLE_DEVICES=$gpu_id python exp_MMA.py --config_path $CONFIG_PATH --start_id $start_id --end_id $end_id > logs/MMA/mma_gpu${gpu_id}_${start_id}-${end_id}.log 2>&1 &
    
    # Small delay to prevent potential issues
    sleep 1
done

echo -e "\nAll jobs submitted! Check logs in the logs/MMA/ directory."