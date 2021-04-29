#! bin/bash

python3.7 run_active6cut.py --model_name kirigami_6_active --num_epoch 10 --nsearch 5 --reward_model model/kl_strain_6.pt 
                     
