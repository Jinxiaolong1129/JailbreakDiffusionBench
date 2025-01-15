#!/bin/bash

# Create logs directory
mkdir -p nohup_logs

# echo "Starting no-attack experiments..."

# # GPU 3: stable-diffusion-3.5-medium
# nohup env CUDA_VISIBLE_DEVICES=3 python exp.py --config_path config/no_attack/stable-diffusion-3.5-medium_noattack.yaml > nohup_logs/sd35_medium_noattack.log 2>&1 &
# echo "Started stable-diffusion-3.5-medium on GPU 3"

# # GPU 4: stable-diffusion-xl-base
# nohup env CUDA_VISIBLE_DEVICES=4 python exp.py --config_path config/no_attack/stable-diffusion-xl-base-0.9_noattack.yaml > nohup_logs/sdxl_base_noattack.log 2>&1 &
# echo "Started stable-diffusion-xl-base on GPU 4"

# GPU 5: flux-1-dev
nohup env CUDA_VISIBLE_DEVICES=5 python exp.py --config_path config/no_attack/flux-1-dev_noattack.yaml > nohup_logs/flux_noattack.log 2>&1 &
echo "Started flux-1-dev on GPU 5"

# # GPU 6: proteus
# nohup env CUDA_VISIBLE_DEVICES=6 python exp.py --config_path config/no_attack/proteus_noattack.yaml > nohup_logs/proteus_noattack.log 2>&1 &
# echo "Started proteus on GPU 6"

# # GPU 7: cogview
# nohup env CUDA_VISIBLE_DEVICES=7 python exp.py --config_path config/no_attack/cogview_noattack.yaml > nohup_logs/cogview_noattack.log 2>&1 &
# echo "Started cogview on GPU 7"

# # Print process IDs for monitoring
# echo "No-attack process IDs:"
# ps aux | grep "python exp.py" | grep -v grep

# echo "No-attack experiments launched. Waiting for completion before starting PGJ experiments..."

# # Wait for no-attack experiments to complete
# while pgrep -f "python exp.py" > /dev/null; do
#     sleep 1200  # Check every minute
#     echo "Still running no-attack experiments..."
# done

echo "No-attack experiments completed. Starting PGJ experiments..."

# GPU 3: stable-diffusion-3.5-medium PGJ
nohup env CUDA_VISIBLE_DEVICES=3 python exp.py --config_path config/PGJ/stable-diffusion-3.5-medium_PGJ.yaml > nohup_logs/sd35_medium_pgj.log 2>&1 &
echo "Started stable-diffusion-3.5-medium PGJ on GPU 3"

# GPU 4: stable-diffusion-xl-base PGJ
nohup env CUDA_VISIBLE_DEVICES=4 python exp.py --config_path config/PGJ/stable-diffusion-xl-base-0.9_PGJ.yaml > nohup_logs/sdxl_base_pgj.log 2>&1 &
echo "Started stable-diffusion-xl-base PGJ on GPU 4"

# # GPU 5: flux-1-dev PGJ
# nohup env CUDA_VISIBLE_DEVICES=2 python exp.py --config_path config/PGJ/flux-1-dev_PGJ.yaml > nohup_logs/flux_pgj.log 2>&1 &
# echo "Started flux-1-dev PGJ on GPU 2"

# GPU 6: proteus PGJ
nohup env CUDA_VISIBLE_DEVICES=6 python exp.py --config_path config/PGJ/proteus_PGJ.yaml > nohup_logs/proteus_pgj.log 2>&1 &
echo "Started proteus PGJ on GPU 6"

# GPU 7: cogview PGJ
nohup env CUDA_VISIBLE_DEVICES=7 python exp.py --config_path config/PGJ/cogview_PGJ.yaml > nohup_logs/cogview_pgj.log 2>&1 &
echo "Started cogview PGJ on GPU 7"

# Print process IDs for monitoring
echo "PGJ process IDs:"
ps aux | grep "python exp.py" | grep -v grep

echo "All experiments launched. Check nohup_logs directory for output."