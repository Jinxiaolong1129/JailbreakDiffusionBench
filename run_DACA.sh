#!/bin/bash

echo "No-attack experiments completed. Starting DACA experiments..."

# GPU 3: stable-diffusion-3.5-medium DACA
nohup env CUDA_VISIBLE_DEVICES=3 python exp.py --config_path config/DACA/stable-diffusion-3.5-medium_DACA.yaml > nohup_logs/sd35_medium_DACA.log 2>&1 &
echo "Started stable-diffusion-3.5-medium DACA on GPU 3"

# GPU 4: stable-diffusion-xl-base DACA
nohup env CUDA_VISIBLE_DEVICES=4 python exp.py --config_path config/DACA/stable-diffusion-xl-base-0.9_DACA.yaml > nohup_logs/sdxl_base_DACA.log 2>&1 &
echo "Started stable-diffusion-xl-base DACA on GPU 4"

# GPU 5: flux-1-dev DACA
nohup env CUDA_VISIBLE_DEVICES=5 python exp.py --config_path config/DACA/flux-1-dev_DACA.yaml > nohup_logs/flux_DACA.log 2>&1 &
echo "Started flux-1-dev DACA on GPU 5"

# GPU 6: proteus DACA
nohup env CUDA_VISIBLE_DEVICES=6 python exp.py --config_path config/DACA/proteus_DACA.yaml > nohup_logs/proteus_DACA.log 2>&1 &
echo "Started proteus DACA on GPU 6"

# GPU 7: cogview DACA
nohup env CUDA_VISIBLE_DEVICES=7 python exp.py --config_path config/DACA/cogview_DACA.yaml > nohup_logs/cogview_DACA.log 2>&1 &
echo "Started cogview DACA on GPU 7"

# Print process IDs for monitoring
echo "DACA process IDs:"
ps aux | grep "python exp.py" | grep -v grep

echo "All experiments launched. Check nohup_logs directory for output."