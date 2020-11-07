import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class lstm_stress_strain(nn.Module):
    def __init__(self):
        super(lstm_stress_strain, self).__init__()
        self.d_prob = 0.5
        self.conv1=nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)             #64
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)                        #32
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)  
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)                        #16
        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)  
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)                        #8
        self.flatten_dim=64*8*8
        self.fc1=nn.Linear(self.flatten_dim,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.act = F.relu
        self.lstm = nn.RNN(input_size=1,hidden_size=256,batch_first=True,bidirectional=False)
        self.output = nn.Linear(256,1)
        self.dropout = nn.Dropout(p = self.d_prob)
    
    def init_hidden(self):
        return torch.zeros(1, 64, 256)
    
    def forward(self,input_X,input_S):
        #print("input_X: ",input_X.size())
        x_c1=self.act(self.conv1(input_X))
        #print("x_c1: ",x_c1.size())
        x_p1=self.pool1(x_c1)
        #print("x_p1: ",x_p1.size())
        x_c2=self.act(self.conv2(x_p1))
        #print("x_c2: ",x_c2.size())
        x_p2=self.pool1(x_c2)
        #print("x_p2: ",x_p2.size())
        x_c3=self.act(self.conv3(x_p2))
        #print("x_c3: ",x_c3.size())
        x_p3=self.pool1(x_c3)
        #print("x_p3: ",x_p3.size())
        x_flatten=x_p3.view(-1,self.flatten_dim)
        #print("x_flatten:",x_flatten.size())
        x_f1=self.dropout(self.act(self.fc1(x_flatten)))
        #print("x_f1:",x_f1.size())
        x_f2= self.dropout(self.act(self.fc2(x_f1)))
        #print("x_f2:",x_f2.size())
        self.h0=self.dropout(self.act(self.fc3(x_f2)))
        self.h0= torch.unsqueeze(self.h0,0)
        #print(self.hidden.size(),x_f2.size())
        hidden_hi,self.hidden = self.lstm(input_S,self.h0)
        #print('tt',self.hidden[0].size(),len(self.hidden))
        y_val = self.output(hidden_hi)
        #print("output:",output.size())
        return y_val

class trainNN_LSTM():
    def __init__(self,nn_model,optimizer,batch_size,epoch,log_step,learing_rate,momentum=0.9):
        self.step = 0
        self.nn_model = nn_model
        self.batch_size = batch_size
        self.learing_rate = learing_rate
        self.epoch = epoch
        self.momentum = momentum
        self.log_step = log_step
        self.loss_function = torch.nn.MSELoss()
        self.training_losses = []
        self.tot_test_losses = []
        self.tot_train_losses  = []
        if optimizer == 'Adam':
            self.optimizer =  optim.Adam(self.nn_model.parameters(), lr=self.learing_rate)
        elif optimizer == 'SGD':
            self.optimizer =  optim.SGD(self.nn_model.parameters(), lr = self.learing_rate, momentum=0.9)
    
    def train(self,trainX,train_SX,train_Y,testX,testSX,testY,cal_test_accuracy=True):
        print('-' * 5 + '  Start training  ' + '-' * 5)

        self.nn_model.train()
        num_training = len(trainX)
        for epoch in range(self.epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.batch_size):
                X_ = trainX[i * self.batch_size:(i + 1) * self.batch_size][:]
                XS_ = train_SX[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = train_Y[i * self.batch_size:(i + 1) * self.batch_size]
                self.nn_model.zero_grad()
                y_predicted = self.nn_model.forward(X_,XS_)
                #print(y_predicted.size())
                loss = self.loss_function(y_predicted,Y_) 
                loss.backward()
                self.optimizer.step()
                self.training_losses.append(loss.item())
                if self.step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f' % (self.step, loss.item()))
                self.step +=1
        torch.save(self.nn_model.state_dict(), 'model/stress_strain.pt')
        
    def cal_accuracy(self,testX,testSX,testY):
        n_test = len(testX)
        self.nn_model.eval()
        self.nn_model.zero_grad()
        avg_loss = 0
        count = 0.0
        with torch.no_grad():
            for i in range(n_test // self.batch_size):
                count += 1.0
                X_ = testX[i * self.batch_size:(i + 1) * self.batch_size][:]
                XS_ = testSX[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = testY[i * self.batch_size:(i + 1) * self.batch_size]
                y_predicted = self.nn_model.forward(X_,XS_)
                loss = self.loss_function(y_predicted,Y_)
                avg_loss += loss.item()
        return avg_loss/count

class stress_model():
    def __init__(self,nn_model,batch_size,itype='load',model_path=None):
        self.nn_model = nn_model
        self.batch_size = batch_size
        if itype == 'load':
            self.load_model(model_path)
    
    def load_model(self,model_path):
        self.nn_model.load_state_dict(torch.load(model_path))
    
    def predict_strain(self,X_data,XS_data):
        self.nn_model.eval()
        self.nn_model.zero_grad()
        with torch.no_grad():
            y_predicted = self.nn_model.forward(X_data,XS_data)
        return y_predicted


    
         


                






