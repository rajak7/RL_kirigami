import os
import torch 
from model.neural_network import cnn_strain_model,reward_model
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from args import get_args
from model.c_vae_linear import CVAE
from loaddata import Kirigami_Dataset
from torch.utils import data
from util import save_model,load_model

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
for i in range(100000,n_files,100000):
    f_state_file = 'dataset/6cut_data/6cut_allstate_'+str(i)+'.npy'
    s_file_name.append(f_state_file)
f_state_file = 'dataset/6cut_data/6cut_allstate_'+str(n_files)+'.npy'
s_file_name.append(f_state_file)
print("Total number of folders:",len(s_file_name))

#load the dataset and sample structures for cvae training  
n_structues = args.ncvae_train
n_pick = n_structues // len(s_file_name)
n_shuffle = 1
for count,f_state_file in enumerate(s_file_name):
    print("reading",f_state_file)
    s_data = np.load(f_state_file)
    num_datapoint = s_data.shape[0]
    for ii in range(n_shuffle):
         suffel_data = np.random.permutation(num_datapoint)
         s_data = s_data[suffel_data]
    tot_XX = torch.from_numpy(s_data[:n_pick]).float()
    tot_XX = tot_XX.to(device)
    y_val=true_model.predict_strain(tot_XX)
    y_val = y_val.to(device)
    if count == 0:
       train_X = tot_XX
       y_predict = y_val
    else:
       train_X = torch.cat((train_X,tot_XX),dim=0)
       y_predict = torch.cat((y_predict,y_val),dim=0)

num_datapoint = tot_XX.shape[0]
print("total number of examples: ",num_datapoint)  

n_train_data = train_X.shape[0]
print('cvae training info: ',n_train_data,train_X.shape,y_predict.shape)
data_list = os.path.join(save_path,'data_hist.npy')
np.save(data_list,y_predict.cpu().numpy())

#create dataloader for CVAE training and a CVAE model
train_data = Kirigami_Dataset(structure=train_X.cpu(),strain=y_predict.cpu())
train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)

z_dim=args.z_dim
c_vae_model = CVAE(z_dim=z_dim)
if device.type == 'cuda':
   c_vae_model=c_vae_model.cuda()
   print("cvae model loaded on gpu done")
optimizer=optim.Adam(c_vae_model.parameters())

#start the training 
c_vae_model.train()
nelbo_trj = []
rec_loss_trj = []
kl_loss_trj = []
for epoch in range(args.num_epoch):
    for idx,data in enumerate(train_loader):
        step = idx + n_train_data * epoch
        optimizer.zero_grad()
        xval = data['structure'].to(device)
        cval = data['strain'].to(device)
        nelbo,rec_loss,kl_loss = c_vae_model.negative_elbo_bound(xval,cval)
        nelbo_trj.append(nelbo.item())
        rec_loss_trj.append(rec_loss.item())
        kl_loss_trj.append(kl_loss.item())
        nelbo.backward()
        optimizer.step()
        if step % 100 == 0:
            print(epoch,step,nelbo.item(),rec_loss.item(),kl_loss.item())

#save model and results 
file_path = os.path.join(save_path,'model_latest.pt')
kl_info = os.path.join(save_path,'kl.npy')
rec_info = os.path.join(save_path,'rec.npy')
nelbo_info = os.path.join(save_path,'nelbo.npy')
save_model(c_vae_model,file_path)
np.save(kl_info,np.asarray(kl_loss_trj))
np.save(rec_info,np.asarray(rec_loss_trj))
np.save(nelbo_info,np.asarray(nelbo_trj))
