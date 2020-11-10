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

#read input argument
args=get_args()
print(args)

#load the ground truth base model
print("ground truth model: ",args.reward_model)
t_model=cnn_strain_model()
true_model = reward_model(t_model,batch_size=64,model_path=args.reward_model)

#path to save all training results
save_path = os.path.join('checkpoints',args.model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

#load the dataset and sample structures for cvae training  
print("Dataset folder: ",args.dataset)
tot_XX = np.load(args.dataset)
tot_cut_loc = np.load(args.dataset_loc)
#do random-shuffling of data 
num_datapoint=tot_XX.shape[0]
for ii in range(3):
    suffel_data = np.random.permutation(num_datapoint)
    tot_XX = tot_XX[suffel_data]
    tot_cut_loc=tot_cut_loc[suffel_data]
print("total number of examples: ",num_datapoint)  

#randomly choose 100 examples as starting training point
n_structure = 100
train_X = torch.from_numpy(tot_XX[:n_structure]).float()
train_X_loc = torch.from_numpy(tot_cut_loc[:n_structure])
remain_X = torch.from_numpy(tot_XX[n_structure:]).float()
remain_X_loc = torch.from_numpy(tot_cut_loc[n_structure:])
y_true=true_model.predict_strain(train_X)
n_train_data = train_X.shape[0]
print("initial training data info: ",n_train_data,train_X.shape,y_true.shape,remain_X.shape,train_X_loc.shape,remain_X_loc.shape)
data_list = os.path.join(save_path,'data_hist.npy')
np.save(data_list,y_true.numpy())

#active learning training loop 
mse_loss = torch.nn.MSELoss(reduction='none')
print("Dataset info:",train_X.shape,remain_X.shape)
cur_max_trj = []
cur_guess_trj = []
cur_struct_id = []
explored_trj = []
explored_guess_trj = []
explored_struct_id = []

#create model 
model=cnn_strain_model()
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
            xval = data['structure']
            yval = data['strain']
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
    with torch.no_grad():
        model.eval()
        y_model = model.forward(remain_X)
        y_index = torch.argsort(y_model.reshape(-1),descending=True)
        top_k = y_index[:n_structure]
        rest_id = y_index[n_structure:]
    #find the true value of these top k structures 
    top_k_X = remain_X[top_k]
    top_k_struct_id = remain_X_loc[top_k]
    rest_struct = remain_X[rest_id]
    rest_struct_id = remain_X_loc[rest_id]
    y_top_k = true_model.predict_strain(top_k_X)
    cur_sample_max_id = torch.argmax(y_top_k.reshape(-1))
    with torch.no_grad():
        model.eval()
        cur_guess_max = model.forward(top_k_X[cur_sample_max_id].unsqueeze(0)) #form model
    cur_max = y_top_k[cur_sample_max_id]    #using ground truth
    cur_max_trj.append(cur_max.item())             
    cur_guess_trj.append(cur_guess_max.item())
    cur_struct_id.append(top_k_struct_id[cur_sample_max_id].numpy())
    print("cur guessed max: ",cur_guess_max,"true val: ",cur_max)
    #update dataset
    remain_X = rest_struct
    remain_X_loc = rest_struct_id
    train_X = torch.cat((train_X,top_k_X),dim=0)
    y_true = torch.cat((y_true,y_top_k),dim=0)
    train_X_loc = torch.cat((train_X_loc,top_k_struct_id),dim=0)
    with torch.no_grad():
        model.eval()
        y_model_ex = model.forward(train_X)
        y_index_max = torch.argmax(y_model_ex.reshape(-1))
        y_true_exp = true_model.predict_strain(train_X[y_index_max].unsqueeze(0))
        explored_guess_trj.append(y_model_ex[y_index_max].item())
        explored_trj.append(y_true_exp.item())
        explored_struct_id.append(train_X_loc[y_index_max].numpy())
    print("Dataset info:",train_X.shape,y_true.shape,remain_X.shape)

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