#!/bin/bash

echo "No-attack experiments completed. Starting DACA experiments..."

# nohup env CUDA_VISIBLE_DEVICES=2 python exp_direct_attack_prompt.py --config_path config/DACA/stable-diffusion-3.5-medium_DACA.yaml > nohup_logs/sd35_medium_DACA.log 2>&1 &
# echo "Started stable-diffusion-3.5-medium DACA on GPU 3"

# nohup env CUDA_VISIBLE_DEVICES=0 python exp_direct_attack_prompt.py --config_path config/DACA/stable-diffusion-xl-base-0.9_DACA.yaml > nohup_logs/sdxl_base_DACA.log 2>&1 &
# echo "Started stable-diffusion-xl-base DACA"




nohup env CUDA_VISIBLE_DEVICES=6 python exp_direct_attack_prompt.py --config_path config/DACA/flux-1-dev_DACA.yaml > nohup_logs/flux_DACA.log 2>&1 &
echo "Started flux-1-dev DACA"

nohup env CUDA_VISIBLE_DEVICES=7 python exp_direct_attack_prompt.py --config_path config/DACA/flux-1-dev_DACA-79.yaml > nohup_logs/flux_DACA-79.log 2>&1 &
echo "Started flux-1-dev DACA"





# nohup env CUDA_VISIBLE_DEVICES=4 python exp_direct_attack_prompt.py --config_path config/DACA/proteus_DACA.yaml > nohup_logs/proteus_DACA.log 2>&1 &
# echo "Started proteus DACA"

# nohup env CUDA_VISIBLE_DEVICES=5 python exp_direct_attack_prompt.py --config_path config/DACA/cogview_DACA.yaml > nohup_logs/cogview_DACA.log 2>&1 &
# echo "Started cogview DACA"

echo "DACA process IDs:"
ps aux | grep "python exp_direct_attack_prompt.py" | grep -v grep

echo "All experiments launched. Check nohup_logs directory for output."