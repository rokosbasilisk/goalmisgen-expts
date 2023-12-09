python train.py --exp_name maze1 --env_name maze_aisc --num_levels 100000 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 1080
python render.py --exp_name maze1_test --env_name maze --distribution_mode hard --param_name hard-500  --model_file PATH_TO_MODEL_FILE
