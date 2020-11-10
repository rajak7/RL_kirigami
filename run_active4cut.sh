#! bin/bash

python3.6 run_active.py --model_name kirigami_4_active --num_epoch 300 --nsearch 10 --reward_model model/kl_strain.pt --dataset dataset/4cut_allstate.npy 
                     
