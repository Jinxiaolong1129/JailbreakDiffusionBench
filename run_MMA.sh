#!/bin/bash

echo "No-attack experiments completed. Starting MMA experiments..."

nohup env CUDA_VISIBLE_DEVICES=2 python exp.py --config_path config/MMA/stable-diffusion-3.5-medium_MMA.yaml > nohup_logs/sd35_medium_MMA.log 2>&1 &
echo "Started stable-diffusion-3.5-medium MMA on GPU 3"



nohup env CUDA_VISIBLE_DEVICES=3 python exp.py --config_path config/MMA/stable-diffusion-xl-base-0.9_MMA.yaml > nohup_logs/sdxl_base_MMA.log 2>&1 &
echo "Started stable-diffusion-xl-base MMA"



nohup env CUDA_VISIBLE_DEVICES=5 python exp.py --config_path config/MMA/flux-1-dev_MMA.yaml > nohup_logs/flux_MMA.log 2>&1 &
echo "Started flux-1-dev MMA"



nohup env CUDA_VISIBLE_DEVICES=7 python exp.py --config_path config/MMA/proteus_MMA.yaml > nohup_logs/proteus_MMA.log 2>&1 &
echo "Started proteus MMA"




echo "MMA process IDs:"
ps aux | grep "python exp.py" | grep -v grep

echo "All experiments launched. Check nohup_logs directory for output."