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
#do random-shuffling of data 
num_datapoint=tot_XX.shape[0]
for ii in range(1):
    suffel_data = np.random.permutation(num_datapoint)
    tot_XX = tot_XX[suffel_data]
print("total number of examples: ",num_datapoint)  
#prepare dataset for C-VAE Training 
train_X = torch.from_numpy(tot_XX[:args.ncvae_train]).float()
y_predict=true_model.predict_strain(train_X)
n_train_data = train_X.shape[0]
print('cvae training info: ',n_train_data,train_X.shape,y_predict.shape)
data_list = os.path.join(save_path,'data_hist.npy')
np.save(data_list,y_predict.numpy())

#create dataloader for CVAE training and a CVAE model
train_data = Kirigami_Dataset(structure=train_X,strain=y_predict)
train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)

z_dim=args.z_dim
c_vae_model = CVAE(z_dim=z_dim)
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
        xval = data['structure']
        cval = data['strain']
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