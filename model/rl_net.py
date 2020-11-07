import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','strain','y_loc'))

class ReplayMemory(object):

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition( *args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQ_net(nn.Module):
    def __init__(self,nactions=13):
        super(DQ_net,self).__init__()
        self.d_prob = 0.5
        self.nactions = nactions
        self.conv1=nn.Conv2d(2,16,kernel_size=3,stride=1,padding=1)             #64
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)                        #32
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)                        #16
        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)                        #8
        self.flatten_dim=64*8*8
        self.fc1=nn.Linear(self.flatten_dim,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,self.nactions)
        self.dropout = nn.Dropout(p = self.d_prob)
        self.act = F.relu

    def forward(self,input_X):
        x_c1=self.act(self.conv1(input_X))
        x_p1=self.pool1(x_c1)
        x_c2=self.act(self.conv2(x_p1))
        x_p2=self.pool1(x_c2)
        x_c3=self.act(self.conv3(x_p2))
        x_p3=self.pool1(x_c3)
        x_flatten=x_p3.view(-1,self.flatten_dim)
        x_f1=self.dropout(self.act(self.fc1(x_flatten)))
        self.x_f2= self.dropout(self.act(self.fc2(x_f1)))
        output=self.fc3(self.x_f2)
        return output

class dqn_model(object):
    def __init__(self,p_net,t_net,memory,n_actions,args):
        self.BATCH_SIZE = args.batch_size  
        self.GAMMA = args.gamma
        self.EPS_START = args.esp_start
        self.EPS_END = args.esp_end
        self.EPS_DECAY = args.esp_decay
        self.TARGET_UPDATE = args.targer_update
        self.policy_net = p_net
        self.target_net = t_net
        self.n_actions = n_actions
        self.optimizer =optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = memory
        self.eps_threshold = 0.0
        
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #print("Model updated")
    
    def predict_all_Q_estimate(self,state,action,next_state,reward):
        all_PQ = self.policy_net(state)
        predict_Q = all_PQ.gather(1,action)
        all_TQ = 0.0
        target_Q = 0.0
        if next_state is not None:
            all_TQ = self.target_net(next_state.float()).detach()
            target_Q = all_TQ.max(1)[0].detach().item()
            all_TQ = all_TQ.data.numpy()
        tot = reward.item()+self.GAMMA*target_Q
        return predict_Q.item(),tot
    
    def predict_Q(self,state,action):
        all_qval = self.policy_net(state)
        predict_Q = all_qval.gather(1,action)
        return predict_Q.item()
        

    def select_action(self,state,current_step):
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * current_step / self.EPS_DECAY)
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)
    
    def optimize_model(self):

        self.optimizer.zero_grad()
        
        if len(self.memory) < self.BATCH_SIZE:
            print("Not enough element in the memory",len(self.memory))
            return
        
        transitions =self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_next_states = non_final_next_states.float()
        
        state_batch = torch.cat(batch.state).float()
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).float()
        strain_batch = torch.cat(batch.strain)
        y_batch = torch.cat(batch.y_loc)
        
        #Compute Q(s,a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = torch.unsqueeze(next_state_values,dim = 1)
        
        #Compute the expected Q value
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        #Compute Loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        #loss = torch.nn.MSELoss(state_action_values, expected_state_action_values.unsqueeze(1))

        #Optimize the model
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1.0, 1.0)
        self.optimizer.step()
        with torch.no_grad():
            tot_Q = torch.mean(state_action_values).detach()
        return loss.item(),tot_Q.item()
        
