#!/bin/bash

mkdir -p nohup_logs


# GPU 0: stable-diffusion-3.5-medium
nohup env CUDA_VISIBLE_DEVICES=0 python exp.py --config_path config/no_attack/stable-diffusion-3.5-medium_noattack.yaml > nohup_logs/sd35_medium_noattack.log 2>&1 &
echo "Started stable-diffusion-3.5-medium on GPU 0"

# GPU 1: stable-diffusion-xl-base
nohup env CUDA_VISIBLE_DEVICES=1 python exp.py --config_path config/no_attack/stable-diffusion-xl-base-0.9_noattack.yaml > nohup_logs/sdxl_base_noattack.log 2>&1 &
echo "Started stable-diffusion-xl-base on GPU 1"

# GPU 2: flux-1-dev
nohup env CUDA_VISIBLE_DEVICES=2 python exp.py --config_path config/no_attack/flux-1-dev_noattack.yaml > nohup_logs/flux_noattack.log 2>&1 &
echo "Started flux-1-dev on GPU 2"

# GPU 3: proteus
nohup env CUDA_VISIBLE_DEVICES=3 python exp.py --config_path config/no_attack/proteus_noattack.yaml > nohup_logs/proteus_noattack.log 2>&1 &
echo "Started proteus on GPU 3"

# GPU 4: cogview
nohup env CUDA_VISIBLE_DEVICES=4 python exp.py --config_path config/no_attack/cogview_noattack.yaml > nohup_logs/cogview_noattack.log 2>&1 &
echo "Started cogview on GPU 4"

# Print process IDs for monitoring
echo "Process IDs:"
ps aux | grep "python exp.py" | grep -v grep

echo "All experiments launched. Check nohup_logs directory for output."


# CUDA_VISIBLE_DEVICES=7 python exp.py --config_path config/PGJ/stable-diffusion-xl-base-0.9_PGJ.yaml