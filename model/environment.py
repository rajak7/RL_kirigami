import torch
import numpy as np 
import random


class generate_state():
    def __init__(self,Lx=200.0,Ly=200.0,ngrids=64,ny=4,lengths=[60.0,100.0,140.0]):
        self.n_actions = 13
        self.Lx = Lx
        self.Ly = Ly
        self.cut_width = 10.0
        self.cut_width_half = 0.5 *  self.cut_width
        self.ngrids = ngrids
        self.del_X = Lx/float(ngrids)          # grid size of the input kirigami tensor in x
        self.del_Y = Ly/float(ngrids)          # grid size of the input kirigami tensor in y
        self.nx = 4                            # number of grids in x direction
        self.ny = ny                            # number of grids in y directionn
        self.del_cx = self.Lx/float(self.nx)   # seperation between two cuts in x direction
        self.del_cy = self.Ly/float(self.ny)   # seperation between two cuts in y direction
        self.action_val(lengths)
    
    def action_val(self,lengths):
        self.action = dict()
        self.action[0] = [0.0,0.0,0.0]
        count = 0
        for i in range(self.nx):
            for l in lengths:
                count += 1
                temp = 0.5*self.del_cx+i*self.del_cx
                self.action[count] = [temp,l,temp+l]        #[start location of cut, length of cut, end location of cut]
    
    def generate_struct(self,yloc,action_num,new_state):
        new_state[1,:,:] = yloc / self.Ly
        for ii in range(self.ngrids):
            x_pos = ii * self.del_X
            for jj in range(self.ngrids):
                y_pos = jj * self.del_Y
                flag = self.cal_flag(x_pos,y_pos,yloc,action_num)
                if flag == True:
                    new_state[0,ii,jj]=1.0
    
    def init_state(self,action_num=0,S0_State=False):
        if S0_State == False:
            yloc = 0.5*self.del_cy
        else:
            action_num = 0
            yloc = 0.0
        new_state = np.zeros((2,self.ngrids,self.ngrids),dtype=float)
        self.generate_struct(yloc,action_num,new_state)
        state_created = krigami_state(new_state,yloc,action_num)
        return state_created
    
    def generate_next_state(self,pre_state,action_num):
        new_state = np.zeros((2,self.ngrids,self.ngrids),dtype=float)
        yloc = pre_state.y_loc + self.del_cy
        #copy information of previous state
        for ii in range(self.ngrids):
            for jj in range(self.ngrids):
                new_state[0,ii,jj] = pre_state.feature[0,ii,jj]
        self.generate_struct(yloc,action_num,new_state)
        state_created = krigami_state(new_state,yloc,action_num)
        return state_created
    
    #genrate a binaay array for a specific kirigami cut
    def create_n_cut_structure(self,action_seq):
        new_state = self.init_state(action_seq[0],S0_State=False)
        for i in range(1,self.ny):
                new_state = self.generate_next_state(new_state,action_seq[i])
        k_structure = new_state.feature[0,:,:]
        k_structure = np.expand_dims(np.expand_dims(k_structure,axis = 0),axis = 0)
        return k_structure

    #only for 4 cut system
    def generate_all_states_4cut(self,tot_states=100):
        count=0
        train_XX = []
        cut_id = []
        for ii in range(self.n_actions):
            y1 = 0.5*self.del_cy
            for jj in range(self.n_actions):
                y2 = y1 + self.del_cy
                for kk in range(self.n_actions):
                    y3 = y2 + self.del_cy
                    for ll in range(self.n_actions):
                        y4 = y3 + self.del_cy
                        if tot_states!= None and count >= tot_states: break
                        count += 1
                        new_state = np.zeros((2,self.ngrids,self.ngrids),dtype=float)
                        self.generate_struct(y1,ii,new_state)  # first layer
                        self.generate_struct(y2,jj,new_state)  # second layer
                        self.generate_struct(y3,kk,new_state)  # third layer
                        self.generate_struct(y4,ll,new_state)  # fourth layer
                        k_structure = new_state[0,:,:]
                        k_structure = np.expand_dims(k_structure,axis = 0)
                        train_XX.append(k_structure)
                        cut_id.append([ii,jj,kk,ll])
                        if count % 100 == 0:
                            print("created: ",count)
        train_XX = np.asarray(train_XX)
        cut_id = np.asarray(cut_id)
        return train_XX,cut_id
    
    #create entire state space for n cut system
    def generate_all_states_6cut(self,tot_states=100):
        count=0
        train_XX = []
        cut_id = []
        for ii in range(self.n_actions):
            y1 = 0.5*self.del_cy
            for jj in range(self.n_actions):
                y2 = y1 + self.del_cy
                for kk in range(self.n_actions):
                    y3 = y2 + self.del_cy
                    for ll in range(self.n_actions):
                        y4 = y3 + self.del_cy
                        for mm in range(self.n_actions):
                            y5 = y4 + self.del_cy
                            for nn in range(self.n_actions):
                                y6 = y5 + self.del_cy
                                if tot_states!= None and count >= tot_states: break
                                count += 1
                                new_state = np.zeros((2,self.ngrids,self.ngrids),dtype=float)
                                self.generate_struct(y1,ii,new_state)  # first layer
                                self.generate_struct(y2,jj,new_state)  # second layer
                                self.generate_struct(y3,kk,new_state)  # third layer
                                self.generate_struct(y4,ll,new_state)  # fourth layer
                                self.generate_struct(y5,ll,new_state)  # fifth layer
                                self.generate_struct(y6,ll,new_state)  # sixth layer
                                k_structure = new_state[0,:,:]
                                k_structure = np.expand_dims(k_structure,axis = 0)
                                train_XX.append(k_structure)
                                cut_id.append([ii,jj,kk,ll,mm,nn])
                                if count % 1000 == 0:
                                    print("created: ",count)
        train_XX = np.asarray(train_XX)
        cut_id = np.asarray(cut_id)
        return train_XX,cut_id

    def cal_flag(self,x_pos,y_pos,yloc,action_num):
        if self.action[action_num][2] < self.Lx:      #end location of cut is within the box
            r_val = (abs(y_pos - yloc) < self.cut_width_half) and (x_pos > self.action[action_num][0]) and (x_pos < self.action[action_num][2])
            return r_val
        else:
            if abs(y_pos - yloc) > self.cut_width_half:       #if point lies outside the ylocation range of cut
                return False
            else:
                if (x_pos + self.Lx > self.action[action_num][2]) and x_pos < self.action[action_num][0]:    #apply pbc
                    return False
                return True

class krigami_state():
    def __init__(self,feature,y_loc,action_num):
        self.feature = feature  #shape(1,2,ngrids,ngrids)
        self.rl_feature = torch.from_numpy(np.expand_dims(feature,axis = 0))
        self.y_loc = y_loc
        self.action_num = action_num
        self.seq_val = 1
        self.strain_val = None
        self.reward_val = None
        self.parent = None
        self.child = None

    def set_parent(self,parent):
        self.parent = parent
        parent.child = self
        self.seq_val = parent.seq_val + 1
    

class environment():
    def __init__(self,r_model,env_model,total_action,args):
        self.r_model = r_model
        self.env_model = env_model
        self.total_action = total_action
        self.seq_len = args.seq_len
        self.threshold = args.threshold
        self.reward_sclae = args.reward_scale

    
    def sample_seq(self,cur_state,action,is_init=False,isterminal=False):
        if is_init == True:
            next_state = self.init_observation(a_num=action,S0=False)
            next_state.set_parent(cur_state)
        elif is_init == False and isterminal == False:
            next_state = self.next_observation(cur_state,action)
        elif is_init == False and isterminal == True:
            next_state = None
        else:
            print("undefied state")
            exit(1)
        
        if isterminal == False:
            cur_state.strain_val = self.cal_reward_next(cur_state)
            cur_state.reward_val = torch.tensor([[0.0]],dtype=torch.float) 
            a_num = torch.tensor([[next_state.action_num]],dtype=torch.long)
        else:
            a_num = torch.tensor([[0]],dtype=torch.long)
            cur_state.strain_val = self.cal_reward(cur_state)
            if cur_state.strain_val <= self.threshold:
                cur_state.reward_val = torch.tensor([[0.0]],dtype=torch.float)
            else:
                cur_state.reward_val = torch.tensor([[cur_state.strain_val/self.reward_sclae]],dtype=torch.float)

        if next_state is not None:
            nn = cur_state.child.rl_feature
        else:
            nn = next_state
        aa = torch.tensor([[cur_state.strain_val]],dtype=torch.float)
        bb = torch.tensor([[cur_state.y_loc]],dtype=torch.float)
        temp=[cur_state.rl_feature,a_num,nn,cur_state.reward_val,aa,bb]        
        return temp,next_state
          
    def next_observation(self,cur_state,action_num):
        next_state = self.env_model.generate_next_state(cur_state,action_num)
        next_state.set_parent(cur_state)
        return next_state
    
    def init_observation(self,a_num=0,S0=False):
        next_state = self.env_model.init_state(action_num=a_num,S0_State=S0)
        return next_state
    
    def cal_reward(self,input_state):
        aa = input_state.feature[0,:,:]
        aa = np.expand_dims(np.expand_dims(aa,axis = 0),axis = 0)
        input_XX= torch.from_numpy(aa).float()
        s_val = self.r_model.predict_strain(input_XX).item()
        return s_val
    
    def cal_reward_next(self,input_state):
        if input_state.child == None:
            aa = input_state
        else:
            aa = input_state.child
        s_val = self.cal_reward(aa)
        return s_val




