#! bin/bash

python3.7 run_DQN.py --buff_size 10000  --n_trj 10000 \
                     --seq_len 7 --num_episode 6000 \
                     --model_name 01run --threshold 15.0 --reward_scale 5.0 \
                     --reward_model model/kl_strain_6.pt \
                     --ny 6 --l1 50.0 --l2 100.0 --l3 150.0 \
                     --esp_start 0.90 --esp_end 0.10 --esp_decay 500 \
                     --gamma 0.90
                     
