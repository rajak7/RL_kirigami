import torch
import os
import numpy as np 
import math
import random

#save DQN models
def save_dqn_model(rl_model,epoch,model_name):
    save_dir = os.path.join('checkpoints',model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model_latest.pt')
    d = {
        'policy_net':  rl_model.policy_net.state_dict(),
        'target_net': rl_model.target_net.state_dict(),
        'replay_memory': rl_model.memory,
        'epoch': epoch,
        'eps_threshold' : rl_model.eps_threshold
    }
    torch.save(d,file_path)

#load DQN model
def load_dqn_model(rl_model,model_name,strict=True,device=None):
    path = os.path.join('checkpoints',model_name,'model_latest.pt')
    ckpt = torch.load(path,map_location=device)
    epoch = ckpt['epoch']
    rl_model.policy_net.load_state_dict(ckpt['policy_net'],strict=strict)
    rl_model.target_net.load_state_dict(ckpt['target_net'],strict=strict)
    rl_model.eps_threshold = ckpt['eps_threshold']
    if args.resume_buffer:
        rl_model.memory = ckpt['replay_memory']
    return epoch

#sample structure using an RL agent
def sample_structure(env,rl_model,seq_len,total_action,cur_step=10000,first_cur_random=True,first_cur_fixed=None):
    cur_state = env.init_observation(S0=True)
    temp_buff = []
    act_taken=[]
    for t in range(seq_len):
        action = rl_model.select_action(cur_state.rl_feature.float(),cur_step).item()
        if first_cur_random == True and t == 0:
            action = random.sample(total_action,1)[0]
        if first_cur_fixed != None and t == 0:
            action = first_cur_fixed
        if t == 0:
            val,next_state = env.sample_seq(cur_state,action=action,is_init=True,isterminal=False)
        elif t > 0 and t < seq_len - 1:
            val,next_state = env.sample_seq(cur_state,action=action,is_init=False,isterminal=False)
        else:
            val,next_state = env.sample_seq(cur_state,action=action,is_init=False,isterminal=True)
        temp_buff.append(val)
        act_taken.append(val[1].item())
        if t < seq_len-1:
            cur_state = next_state
    return cur_state,val,temp_buff,act_taken

#sample a random structure
def sample_random_structure(env,seq_len,total_action):
    cur_state = env.init_observation(S0=True)
    temp_buff = []
    act_taken=[]
    for t in range(seq_len):
        action = random.sample(total_action,1)[0]
        if t == 0:
            val,next_state = env.sample_seq(cur_state,action=action,is_init=True,isterminal=False)
        elif t > 0 and t < seq_len - 1:
            val,next_state = env.sample_seq(cur_state,action=action,is_init=False,isterminal=False)
        else:
            val,next_state = env.sample_seq(cur_state,action=action,is_init=False,isterminal=True)
        temp_buff.append(val)
        act_taken.append(val[1].item())
        if t < seq_len-1:
            cur_state = next_state
    return cur_state,val,temp_buff,act_taken

#sample a batch of structures from RL agent to compute expected reward and Q value of terminal state
def estimate_random_policy(env,seq_len,total_action,samples=64):
    r_trj = []
    str_trj = []
    for i in range(samples):
        cur_state, cur_state_info,_,_ = sample_random_structure(env,seq_len,total_action)
        r_trj.append(cur_state.reward_val)
        str_trj.append(cur_state.strain_val)
    r_trj = np.asarray(r_trj)
    str_trj = np.asarray(str_trj)
    return r_trj,str_trj

#sample a batch of structures from RL agent to compute expected reward and Q value of terminal state
def estimate_policy_reward(env,rl_model,seq_len,total_action,cur_step=10000,first_cur_random=True,samples=64):
    q_trj = []
    r_trj = []
    str_trj = []
    for i in range(samples):
        cur_state, cur_state_info,_,_ = sample_structure(env,rl_model,seq_len,total_action,cur_step,first_cur_random)
        q_trj.append(rl_model.predict_Q(cur_state_info[0].float(),cur_state_info[1]))
        r_trj.append(cur_state.reward_val)
        str_trj.append(cur_state.strain_val)
    q_trj = np.asarray(q_trj)
    r_trj = np.asarray(r_trj)
    str_trj = np.asarray(str_trj)
    return q_trj,r_trj,str_trj

#sample the hidden feature from the model from time=3
def viz_hidden_3(env,rl_model,seq_len,action_seq,cur_step=10000):
    cur_state = env.init_observation(S0=True)
    q_seq = [] 
    hh = []
    cut_loc = []
    for row1 in action_seq:
        val,row1_state = env.sample_seq(cur_state,action=row1,is_init=True,isterminal=False) #adds 1st cut
        for row2 in action_seq:
            val2,row2_state = env.sample_seq(row1_state,action=row2,is_init=False,isterminal=False) #adds 2nd cut
            for row3 in action_seq: 
                val3,row3_state = env.sample_seq(row2_state,action=row3,is_init=False,isterminal=False) #adds 3rd cut
                #follow the policy and sample the next best action
                with torch.no_grad():
                    action = rl_model.policy_net(row3_state.rl_feature.float()).max(1)[1].view(1, 1)
                    hidden_X = rl_model.policy_net.x_f2.detach().view(-1).numpy()
                    Q_val = rl_model.predict_Q(row3_state.rl_feature.float(),action)
                    q_seq.append(Q_val)
                    hh.append(hidden_X)
                    cut_loc.append([row1,row2,row3,action.item()])
    hh = np.asarray(hh) 
    return hh,q_seq,cut_loc

#sample the hidden feature from the model from time=2
def viz_hidden_2(env,rl_model,seq_len,action_seq,cur_step=10000):
    cur_state = env.init_observation(S0=True)
    q_seq = [] 
    hh = []
    cut_loc = []
    for row1 in action_seq:
        val,row1_state = env.sample_seq(cur_state,action=row1,is_init=True,isterminal=False) #adds 1st cut
        for row2 in action_seq:
            val2,row2_state = env.sample_seq(row1_state,action=row2,is_init=False,isterminal=False) #adds 2nd cut
            #follow the policy and sample the next best action
            with torch.no_grad():
                action = rl_model.policy_net(row2_state.rl_feature.float()).max(1)[1].view(1, 1)
                hidden_X = rl_model.policy_net.x_f2.detach().view(-1).numpy()
                Q_val = rl_model.predict_Q(row2_state.rl_feature.float(),action)
                q_seq.append(Q_val)
                hh.append(hidden_X)
                cut_loc.append([row1,row2,action.item(),None])
    hh = np.asarray(hh) 
    return hh,q_seq,cut_loc

#sample the hidden feature from the model from time=4
def viz_hidden_4(env,rl_model,seq_len,action_seq,cur_step=10000):
    cur_state = env.init_observation(S0=True)
    q_seq = [] 
    hh = []
    cut_loc = []
    for row1 in action_seq:
        val,row1_state = env.sample_seq(cur_state,action=row1,is_init=True,isterminal=False) #adds 1st cut
        for row2 in action_seq:
            val2,row2_state = env.sample_seq(row1_state,action=row2,is_init=False,isterminal=False) #adds 2nd cut
            for row3 in action_seq: 
                val3,row3_state = env.sample_seq(row2_state,action=row3,is_init=False,isterminal=False) #adds 3rd cut
                for row4 in action_seq:
                    val4,row4_state = env.sample_seq(row3_state,action=row4,is_init=False,isterminal=False) #adds 4rd cut
                    #follow the policy and sample the next best action
                    with torch.no_grad():
                        if seq_len > 5:
                            action = rl_model.policy_net(row4_state.rl_feature.float()).max(1)[1].view(1, 1)
                        else:
                            action = torch.tensor([[0]],dtype=torch.long)
                            _ = rl_model.policy_net(row4_state.rl_feature.float())
                        hidden_X = rl_model.policy_net.x_f2.detach().view(-1).numpy()
                        Q_val = rl_model.predict_Q(row4_state.rl_feature.float(),action)
                        q_seq.append(Q_val)
                        hh.append(hidden_X)
                        cut_loc.append([row1,row2,row3,action.item()])
    hh = np.asarray(hh) 
    return hh,q_seq,cut_loc

