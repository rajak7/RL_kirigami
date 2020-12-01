import os
import torch 
from model.neural_network import cnn_strain_model,trainNN,reward_model
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from args import get_args
from loaddata import Kirigami_Dataset
from torch.utils.data import DataLoader
from util import save_model,load_model
import time

start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#read input argument
args=get_args()
print(args)

#load the ground truth base model
print("ground truth model: ",args.reward_model)
t_model=cnn_strain_model()
if device.type == 'cuda':
   t_model=t_model.cuda()
   print("model loaded on gpu done")
true_model = reward_model(t_model,batch_size=64,model_path=args.reward_model)

#path to save all training results
save_path = os.path.join('checkpoints',args.model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#get all file name info
n_files=4826809
s_file_name = []
l_file_name = []
for i in range(100000,n_files,100000):
    f_state_file = 'dataset/6cut_data/6cut_allstate_'+str(i)+'.npy'
    f_loc_file = 'dataset/6cut_data/6cut_location_'+str(i)+'.npy'
    s_file_name.append(f_state_file)
    l_file_name.append(f_loc_file)
f_state_file = 'dataset/6cut_data/6cut_allstate_'+str(n_files)+'.npy'
f_loc_file = 'dataset/6cut_data/6cut_location_'+str(n_files)+'.npy'
s_file_name.append(f_state_file)
l_file_name.append(f_loc_file)

#load the dataset and sample structures for cvae training  
n_structure = 100
n_pick = n_structure // len(s_file_name)
n_shuffle = 1
print("file per frame",n_pick,n_structure)
#reading structure files 
count = -1
for f_state_file,f_loc_file in zip(s_file_name,l_file_name):
    count +=1
    s_data = np.load(f_state_file)
    s_loc = np.load(f_loc_file)
    num_datapoint = s_data.shape[0]
    for ii in range(n_shuffle):
        suffel_data = np.random.permutation(num_datapoint)
        s_data = s_data[suffel_data]
        s_loc = s_loc[suffel_data]
    if count == 0:
       tot_XX = s_data[:n_pick]
       tot_cut_loc = s_loc[:n_pick]
    else:
       tot_XX = np.concatenate((tot_XX,s_data[:n_pick]),axis=0) 
       tot_cut_loc = np.concatenate((tot_cut_loc,s_loc[:n_pick]),axis=0)

num_datapoint=tot_XX.shape[0]
print("total number of examples: ",num_datapoint)  

train_X = torch.from_numpy(tot_XX).float()
train_X = train_X.to(device)
train_X_loc = torch.from_numpy(tot_cut_loc)
y_true=true_model.predict_strain(train_X)
n_train_data = train_X.shape[0]
print("initial training data info: ",n_train_data,train_X.shape,y_true.shape,train_X_loc.shape)
data_list = os.path.join(save_path,'data_hist.npy')
np.save(data_list,y_true.cpu().numpy())

#active learning training loop 
mse_loss = torch.nn.MSELoss(reduction='none')
print("Dataset info:",train_X.shape)
cur_max_trj = []
cur_guess_trj = []
cur_struct_id = []
explored_trj = []
explored_guess_trj = []
explored_struct_id = []

#create model 
train_X = train_X.cpu()
y_true = y_true.cpu()
model=cnn_strain_model()
if device.type == 'cuda':
   model=model.cuda()
   print("training model loaded on gpu done")
for outer_loop in range(args.nsearch):
    #create data loader
    train_data = Kirigami_Dataset(structure=train_X,strain=y_true)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
    #create optimizer
    optimizer=optim.Adam(model.parameters())
    model.train()
    #train the new model
    print("========Training for outer_loop:",outer_loop,"========")
    for epoch in range(args.num_epoch):
        for idx,data in enumerate(train_loader):
            step = idx + n_train_data * epoch
            optimizer.zero_grad()
            xval = data['structure'].to(device)
            yval = data['strain'].to(device)
            y_predict = model.forward(xval)
            loss = mse_loss(input=y_predict,target=yval)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('epoch: ',epoch,step,loss.item())
    #train for outer loop finished
    print("========Training for outer_loop:",outer_loop,"finished ========")
    #find the current guess of top 100 structures by currect model
    ival = -1
    for f_state_file,f_loc_file in zip(s_file_name,l_file_name):
        ival += 1
        print("reading",ival,f_state_file,f_loc_file)
        s_data = np.load(f_state_file)
        s_loc = np.load(f_loc_file)
        remain_X = torch.from_numpy(s_data).float()
        remain_X_loc = torch.from_numpy(s_loc)
        with torch.no_grad():
             model.eval()
             print("predicting for ",remain_X.shape)
             remain_loader = DataLoader(remain_X, batch_size=1024, shuffle=False)
             rval_trj = []
             yval_trj = []
             for rval in remain_loader:
                 rval = rval.to(device)
                 yrval = model.forward(rval)
                 rval = rval.cpu()
                 yrval = yrval.cpu()
                 rval_trj.append(rval)
                 yval_trj.append(yrval)
             remain_X = torch.cat(rval_trj,dim=0)
             y_model_temp = torch.cat(yval_trj,dim=0)
             y_index_chunk = torch.argsort(y_model_temp.reshape(-1),descending=True)
             top_k_chuck = y_index_chunk[:n_structure]
             if ival == 0 :
                remain_X_chunk = remain_X[top_k_chuck]
                y_model_chunk = y_model_temp[top_k_chuck]
                remain_X_loc_chunk = remain_X_loc[top_k_chuck]
             else:
                remain_X_chunk = torch.cat((remain_X_chunk,remain_X[top_k_chuck]),dim=0)
                y_model_chunk =  torch.cat((y_model_chunk,y_model_temp[top_k_chuck]),dim=0)
                remain_X_loc_chunk = torch.cat((remain_X_loc_chunk,remain_X_loc[top_k_chuck]),dim=0)
    #sort the chuck files to get top 100 files
    y_index = torch.argsort(y_model_chunk.reshape(-1),descending=True)
    top_k = y_index[:n_structure]
    top_k_X = remain_X_chunk[top_k]   # top K structures from entire state space
    top_k_struct_id = remain_X_loc_chunk[top_k]
    #find the true value of these top k structures
    top_k_X = top_k_X.to(device) 
    y_top_k = true_model.predict_strain(top_k_X)
    y_top_k = y_top_k.cpu()
    cur_sample_max_id = torch.argmax(y_top_k.reshape(-1))
    with torch.no_grad():
        model.eval()
        cur_guess_max = model.forward(top_k_X[cur_sample_max_id].unsqueeze(0)) #form model
    cur_max = y_top_k[cur_sample_max_id]    #using ground truth
    cur_max_trj.append(cur_max.item())             
    cur_guess_trj.append(cur_guess_max.cpu().item())
    cur_struct_id.append(top_k_struct_id[cur_sample_max_id].numpy())
    print("cur guessed max: ",cur_guess_max,"true val: ",cur_max)
    #update dataset
    top_k_X = top_k_X.cpu()
    train_X = torch.cat((train_X,top_k_X),dim=0)
    y_true = torch.cat((y_true,y_top_k),dim=0)
    print(train_X_loc.shape,top_k_struct_id.shape)
    train_X_loc = torch.cat((train_X_loc,top_k_struct_id),dim=0)
    with torch.no_grad():
        model.eval()
        train_X = train_X.to(device)
        y_model_ex = model.forward(train_X) 
        y_model_ex = y_model_ex.cpu()
        y_index_max = torch.argmax(y_model_ex.reshape(-1))
        y_true_exp = true_model.predict_strain(train_X[y_index_max].unsqueeze(0))
        y_true_exp = y_true_exp.cpu()
        train_X = train_X.cpu()
        explored_guess_trj.append(y_model_ex[y_index_max].item())
        explored_trj.append(y_true_exp.item())
        explored_struct_id.append(train_X_loc[y_index_max].numpy())
    print("Dataset info:",train_X.shape,y_true.shape)

log_path = os.path.join(save_path,'log.txt')
log_file = open(log_path,'w')
print("=====Cur Max info:=====")
for a,b in zip(cur_max_trj,cur_guess_trj):
    print('Cur info: True {0:12.6f} Guess {1:12.6f}'.format(a,b))
    log_file.write('Cur info: True {0:12.6f} Guess {1:12.6f} \n'.format(a,b))


print("=====Explored Max info:=====")
for a,b in zip(explored_trj,explored_guess_trj):
    print('Explored info: True {0:12.6f} Guess {1:12.6f}'.format(a,b))
    log_file.write('Explored info: True {0:12.6f} Guess {1:12.6f} \n'.format(a,b))

#save model and results 
file_path = os.path.join(save_path,'model_latest.pt')
cur_max = os.path.join(save_path,'cur_max.npy')
cur_guess_max = os.path.join(save_path,'cur_guess_max.npy')
cur_max_id = os.path.join(save_path,'cur_max_id.npy')
explored_guess = os.path.join(save_path,'explored_guess.npy')
explored_max = os.path.join(save_path,'explored_max.npy')
explored_max_id = os.path.join(save_path,'explored_max_id.npy')
save_model(model,file_path)
np.save(cur_max,np.asarray(cur_max_trj))
np.save(cur_guess_max,np.asarray(cur_guess_trj))
np.save(cur_max_id,np.asarray(cur_struct_id))
np.save(explored_guess,np.asarray(explored_guess_trj))
np.save(explored_max,np.asarray(explored_trj))
np.save(explored_max_id,np.asarray(explored_struct_id))

print("--- %s train time in  seconds ---" % (time.time() - start_time))
