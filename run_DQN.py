import os
import torch
from args import get_args
from model.environment import *
from model.neural_network import *
from model.rl_net import *
from train import train
from util import load_dqn_model

#read input argument
args=get_args()
print(args)

if args.seq_len != (args.ny + 1):
    raise Exception('seq_len should be ny+1')

s_model=generate_state(args.Lx,args.Ly,args.ngrids,args.ny,lengths=[args.l1,args.l2,args.l3])

kirigami_nn_model=cnn_strain_model()
r_model = reward_model(kirigami_nn_model,batch_size=args.batch_size,model_path=args.reward_model)
total_action = [i for i in range(s_model.n_actions) ]
env = environment(r_model,s_model,total_action,args)
print("total_action:",total_action,"and no. of cuts",args.ny)

#create DQN network 
policy_net = DQ_net(nactions=len(total_action))
target_net = DQ_net(nactions=len(total_action))

if args.train == True:
    if args.resume == False: 
        r_buffer = ReplayMemory(args.buff_size)
        rl_model = dqn_model(policy_net,target_net,r_buffer,len(total_action),args)
    if args.resume == True:
        r_buffer = None
        rl_model = dqn_model(policy_net,target_net,r_buffer,len(total_action),args)
        previous  = os.path.join(args.model_name,'previous')
        epoch = load_dqn_model(rl_model,previous)
        r_buffer = rl_model.memory 
        print("Resuming Training from previous training epoch: ",epoch)
    train(rl_model,env,total_action,r_buffer,args)
else:
    r_buffer = None
    rl_model = dqn_model(policy_net,target_net,r_buffer,len(total_action),args)
    epoch = load_dqn_model(rl_model,args.model_name)
    print("RL model loaded from epoch: ",epoch)
    
