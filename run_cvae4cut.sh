#! bin/bash

python3.6 run_CVAE.py --model_name kirigami_4_cvae --num_epoch 5 --reward_model model/kl_strain.pt --dataset dataset/4cut_allstate.npy --z_dim 10 
                     
