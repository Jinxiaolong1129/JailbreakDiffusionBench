#!/bin/bash

# List of Stable Diffusion models in the discord directory with their corresponding GPUs
declare -A MODEL_TO_GPU
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-v1-5.yaml"]=0
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-2.yaml"]=1
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-xl-base-0.9.yaml"]=2
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-safe.yaml"]=3
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-3-medium.yaml"]=4
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-3.5-medium.yaml"]=5
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-3.5-large.yaml"]=6
MODEL_TO_GPU["config/no_attack/benign/discord/stable-diffusion-3.5-large-turbo.yaml"]=7

# Create logs directory if it doesn't exist
mkdir -p logs/discord

# Function to check if GPU is busy
is_gpu_busy() {
  local gpu_id=$1
  
  # Check if any process is using this GPU
  local processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i $gpu_id)
  
  if [ -z "$processes" ]; then
    # No process using this GPU
    return 1  # False - GPU is not busy
  else
    # There are processes using this GPU
    return 0  # True - GPU is busy
  fi
}

# Function to run a task on a specific GPU
run_task() {
  local MODEL=$1
  local GPU_ID=$2
  local MODEL_NAME=$(basename "${MODEL}" .yaml)
  
  echo "Starting experiment for ${MODEL_NAME} on GPU ${GPU_ID} at $(date)"
  
  # Set CUDA_VISIBLE_DEVICES to use only one specific GPU
  # Run the script with nohup to keep it running after terminal closes
  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python exp_no_attack.py \
    --config_path "${MODEL}" > "logs/discord/${MODEL_NAME}.log" 2>&1 &
  
  echo "Started process $! for ${MODEL_NAME} on GPU ${GPU_ID}"
  
  # Mark this model as running
  RUNNING_MODELS[$MODEL]=$!
}

# Initialize arrays to track status
declare -A RUNNING_MODELS  # Maps model path to PID
declare -A COMPLETED_MODELS  # Maps model path to completion status (1=completed)

# Initialize completed models
for MODEL in "${!MODEL_TO_GPU[@]}"; do
  COMPLETED_MODELS[$MODEL]=0
  RUNNING_MODELS[$MODEL]=0
done

echo "Starting monitoring at $(date). Will check GPU status every 30 seconds."

# Keep checking until all models have completed
while true; do
  all_completed=true
  
  for MODEL in "${!MODEL_TO_GPU[@]}"; do
    # Skip models that are already completed
    if [[ "${COMPLETED_MODELS[$MODEL]}" == "1" ]]; then
      continue
    fi
    
    all_completed=false
    
    # If model is running, check if it's still running
    if [[ "${RUNNING_MODELS[$MODEL]}" != "0" ]]; then
      if ! ps -p ${RUNNING_MODELS[$MODEL]} > /dev/null 2>&1; then
        echo "Experiment for $(basename "${MODEL}" .yaml) completed at $(date)"
        COMPLETED_MODELS[$MODEL]=1
        RUNNING_MODELS[$MODEL]=0
      fi
    else
      # Model is not running yet, check if its GPU is available
      GPU_ID=${MODEL_TO_GPU[$MODEL]}
      
      if ! is_gpu_busy $GPU_ID; then
        # GPU is free, run the model
        run_task "$MODEL" "$GPU_ID"
      else
        echo "GPU $GPU_ID is busy. Cannot start $(basename "${MODEL}" .yaml) yet."
      fi
    fi
  done
  
  # Exit loop if all models completed
  if $all_completed; then
    break
  fi
  
  # Wait before checking again
  sleep 30
done

echo "All experiments completed at $(date). Check logs in logs/discord/ directory."


