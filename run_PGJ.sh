#!/bin/bash

mkdir -p nohup_logs

echo "No-attack experiments completed. Starting PGJ experiments..."

# nohup env CUDA_VISIBLE_DEVICES=2 python exp_direct_attack_prompt.py --config_path config/PGJ/stable-diffusion-3.5-medium_PGJ.yaml > nohup_logs/sd35_medium_pgj.log 2>&1 &
# echo "Started stable-diffusion-3.5-medium PGJ on GPU 3"

nohup env CUDA_VISIBLE_DEVICES=3 python exp_direct_attack_prompt.py --config_path config/PGJ/stable-diffusion-xl-base-0.9_PGJ.yaml > nohup_logs/sdxl_base_pgj.log 2>&1 &
echo "Started stable-diffusion-xl-base PGJ on GPU 4"

nohup env CUDA_VISIBLE_DEVICES=2 python exp_direct_attack_prompt.py --config_path config/PGJ/flux-1-dev_PGJ.yaml > nohup_logs/flux_pgj.log 2>&1 &
echo "Started flux-1-dev PGJ on GPU 2"

nohup env CUDA_VISIBLE_DEVICES=5 python exp_direct_attack_prompt.py --config_path config/PGJ/proteus_PGJ.yaml > nohup_logs/proteus_pgj.log 2>&1 &
echo "Started proteus PGJ on GPU 6"

nohup env CUDA_VISIBLE_DEVICES=7 python exp_direct_attack_prompt.py --config_path config/PGJ/cogview_PGJ.yaml > nohup_logs/cogview_pgj.log 2>&1 &
echo "Started cogview PGJ on GPU 7"

echo "PGJ process IDs:"
ps aux | grep "python exp_direct_attack_prompt.py" | grep -v grep

echo "All experiments launched. Check nohup_logs directory for output."