import os
import torch
import torch.nn as nn
import numpy as np
import math
import random
from util import estimate_policy_reward,save_dqn_model

##create iniital memory r_buffer  for training
def create_buffer_with_sample(env,r_buffer,total_action,args):
    for ii in range(args.n_trj):
        cut_loc = random.sample(total_action,args.seq_len)
        cur_state = env.init_observation(S0=True)
        for jj in range(args.seq_len):
            if jj == 0:
                val,next_state = env.sample_seq(cur_state,action=cut_loc[jj],is_init=True,isterminal=False)
            elif jj > 0 and jj < args.seq_len-1:
                val,next_state = env.sample_seq(cur_state,action=cut_loc[jj],is_init=False,isterminal=False)
            else:
                val,next_state = env.sample_seq(cur_state,action=cut_loc[jj],is_init=False,isterminal=True)
            r_buffer.push(val[0],val[1],val[2],val[3],val[4],val[5])
        #print(cur_state.seq_val,cur_state.y_loc)
        if jj < args.seq_len-1:
            cur_state = next_state
    print("Buffer is filled with random samples:",len(r_buffer))
    return  cur_state

def train(rl_model,env,total_action,r_buffer,args):
    #fill r_buffer with samples
    if args.resume_buffer == False :
        _ = create_buffer_with_sample(env,r_buffer,total_action,args)
    else:
       print("Buffer resumed from previous run as ",r_buffer.__len__(),"samples")
    cur_step = 0
    tot_loss = []
    tot_reward = []
    tot_strain = []
    tot_Q_train = []
    expected_R_mean = []
    expected_R_std = []
    expected_Q_mean = []
    expected_Q_std = []
    path = os.path.join('checkpoints',args.model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    log_file = os.path.join(path,'log.txt')
    log = open(log_file,'w')
    print("=====Training Starts======")
    for i_episode in range(args.num_episode):
        rl_model.target_net.eval()
        rl_model.policy_net.train()
        cur_state = env.init_observation(S0=True)
        for jj in range(args.seq_len):
            cur_step += 1
            action = rl_model.select_action(cur_state.rl_feature.float(),cur_step).item()
            if jj == 0:
                val,next_state = env.sample_seq(cur_state,action=action,is_init=True,isterminal=False)
            elif jj > 0 and jj < args.seq_len-1:
                val,next_state = env.sample_seq(cur_state,action=action,is_init=False,isterminal=False)
            else:
                val,next_state = env.sample_seq(cur_state,action=action,is_init=False,isterminal=True)
            #optimize the model
            loss,train_Q = rl_model.optimize_model()
            #add new sampled data into the r_buffer
            r_buffer.push(val[0],val[1],val[2],val[3],val[4],val[5])
            if jj < args.seq_len-1:
                cur_state = next_state
        # save results of the terminal state
        tot_reward.append(val[3])
        tot_strain.append(val[4])
        tot_loss.append(loss)
        tot_Q_train.append(train_Q)
        #update target and compute expected reward 
        if i_episode % rl_model.TARGET_UPDATE == 0:
            rl_model.update_target()
            rl_model.policy_net.eval()
            q_trj,r_trj = estimate_policy_reward(env,rl_model,args.seq_len,total_action,cur_step,True,args.batch_size)
            q_mean = np.mean(q_trj)
            q_std = np.std(q_trj)
            r_mean = np.mean(r_trj)
            r_std = np.std(r_trj)
            print("episode: {0:4d} step: {1:6d} strain: {2:6.3f} reward: {3:6.3f} loss: {4:6.3f} Q: {5:6.3f} r_mean: {6:6.3f} q_mean: {7:6.3f} epsilon: {8:4.3f}"
            .format(i_episode,cur_step,cur_state.strain_val,cur_state.reward_val.item(),loss,train_Q,r_mean,q_mean,rl_model.eps_threshold))
            log.write("episode: {0:4d} step: {1:6d} strain: {2:6.3f} reward: {3:6.3f} loss: {4:6.3f} Q: {5:6.3f} r_mean: {6:6.3f} q_mean: {7:6.3f} epsilon: {8:4.3f}\n"
            .format(i_episode,cur_step,cur_state.strain_val,cur_state.reward_val.item(),loss,train_Q,r_mean,q_mean,rl_model.eps_threshold))
            expected_R_mean.append(r_mean)
            expected_R_std.append(r_std)
            expected_Q_mean.append(q_mean)
            expected_Q_std.append(q_std)
    #save final model
    save_dqn_model(rl_model,i_episode,args.model_name)
    #save training results 
    np.save(path+'/'+'tot_loss',tot_loss)
    np.save(path+'/'+'tot_reward',tot_reward)
    np.save(path+'/'+'tot_strain',tot_strain)
    np.save(path+'/'+'tot_Q_train',tot_Q_train)
    np.save(path+'/'+'expected_R_mean',expected_R_mean)
    np.save(path+'/'+'expected_R_std',expected_R_std)
    np.save(path+'/'+'expected_Q_mean',expected_Q_mean)
    np.save(path+'/'+'expected_Q_std',expected_Q_std)

            



