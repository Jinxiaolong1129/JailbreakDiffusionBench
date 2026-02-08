# GPU 0: cogview.yaml
mkdir -p logs/sneakprompt-0.6-l2
nohup env CUDA_VISIBLE_DEVICES=0 python exp_sneakprompt.py --config_path config/sneakprompt/cogview.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/cogview.log 2>&1 &

# GPU 1: flux-1-dev.yaml
nohup env CUDA_VISIBLE_DEVICES=1 python exp_sneakprompt.py --config_path config/sneakprompt/flux-1-dev.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/flux_1_dev.log 2>&1 &

# GPU 2: hunyuan-dit-v1.2-distilled.yaml
nohup env CUDA_VISIBLE_DEVICES=2 python exp_sneakprompt.py --config_path config/sneakprompt/hunyuan-dit-v1.2-distilled.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/hunyuan_dit_v1_2_distilled.log 2>&1 &

# GPU 3: stable-diffusion-3-medium.yaml
nohup env CUDA_VISIBLE_DEVICES=3 python exp_sneakprompt.py --config_path config/sneakprompt/stable-diffusion-3-medium.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/stable_diffusion_3_medium.log 2>&1 &

# GPU 4: stable-diffusion-3.5-large-turbo.yaml
nohup env CUDA_VISIBLE_DEVICES=4 python exp_sneakprompt.py --config_path config/sneakprompt/stable-diffusion-3.5-large-turbo.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/stable_diffusion_3_5_large_turbo.log 2>&1 &

# GPU 5: stable-diffusion-3.5-medium.yaml
nohup env CUDA_VISIBLE_DEVICES=5 python exp_sneakprompt.py --config_path config/sneakprompt/stable-diffusion-3.5-medium.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/stable_diffusion_3_5_medium.log 2>&1 &

# GPU 6: stable-diffusion-v1-5.yaml
nohup env CUDA_VISIBLE_DEVICES=6 python exp_sneakprompt.py --config_path config/sneakprompt/stable-diffusion-v1-5.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/stable_diffusion_v1_5.log 2>&1 &

# GPU 7: stable-diffusion-xl-base-0.9.yaml
nohup env CUDA_VISIBLE_DEVICES=7 python exp_sneakprompt.py --config_path config/sneakprompt/stable-diffusion-xl-base-0.9.yaml --start_id 1 --end_id 400 > logs/sneakprompt-0.6-l2/stable_diffusion_xl_base_0_9.log 2>&1 &


# Commands to check if the processes are running
echo "To check if all processes are running, use:"
echo "ps aux | grep exp_sneakprompt.py"

# Commands to check GPU usage
echo "To check GPU usage, use:"
echo "nvidia-smi"

# Commands to check the logs
echo "To check the last 10 lines of all logs, use:"
echo "tail -n 10 logs/sneakprompt-0.6-l2/*.log"