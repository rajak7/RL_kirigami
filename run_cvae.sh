#! bin/bash

python3.7 run_CVAE.py --model_name kirigami_6_cvae_z_12 --num_epoch 500 --reward_model model/kl_strain_6.pt  --z_dim 12  --ncvae_train 70000 
                     
