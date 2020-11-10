import torch
import torch.nn as nn
import torch.nn.functional as F

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

class encoder(nn.Module):
    def __init__(self,d_prob=0.5,z_dim=10):
        super(encoder, self).__init__()
        self.d_prob = d_prob
        self.z_dim = z_dim
        self.conv1=nn.Conv2d(2,16,kernel_size=3,stride=1,padding=1)        #64   
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2)                   #32
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)     
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)                   #16 
        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)    
        self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)                   #8
        self.flatten_dim=64*8*8
        self.fc1=nn.Linear(self.flatten_dim,1024)
        self.dropout1 = nn.Dropout(p = self.d_prob)
        self.fc2=nn.Linear(1024,512)
        self.dropout2 = nn.Dropout(p = self.d_prob)
        self.fc3=nn.Linear(512,2*self.z_dim)
        self.act = F.relu

    def forward(self,X,Y):
        c_val=torch.ones(*X.shape) * Y.reshape(-1,1,1,1)
        input_XY = torch.cat((X,c_val),dim=1)
        #print(c_val.shape,input_XY.shape)
        x_c1=self.act(self.conv1(input_XY))
        x_p1=self.pool1(x_c1)
        x_c2=self.act(self.conv2(x_p1))
        x_p2=self.pool1(x_c2)
        x_c3=self.act(self.conv3(x_p2))
        x_p3=self.pool1(x_c3)
        x_flatten=x_p3.view(-1,self.flatten_dim)
        x_f1=self.dropout1(self.act(self.fc1(x_flatten)))
        self.x_f2= self.dropout2(self.act(self.fc2(x_f1)))
        h=self.fc3(self.x_f2)
        m,v = gaussian_parameters(h,dim=1)
        return m,v

class decoder(nn.Module):
    def __init__(self,d_prob=0.5,z_dim=10,c_dim=1):
        super(decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.d_prob = d_prob
        self.flatten_dim=64*8*8
        self.fc1=nn.Linear(self.z_dim+self.c_dim,512)
        self.dropout1 = nn.Dropout(p = self.d_prob)
        self.fc2=nn.Linear(512,1024)
        self.dropout2 = nn.Dropout(p = self.d_prob)
        self.fc3=nn.Linear(1024,self.flatten_dim)
        self.dconv1 = nn.ConvTranspose2d(64, 32,kernel_size=4,stride=2,padding=1, bias=False)   #16
        self.bn1 = nn.BatchNorm2d(32)
        self.dconv2 = nn.ConvTranspose2d(32, 16,kernel_size=4,stride=2,padding=1, bias=False)   #32
        self.bn2 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(16, 1,kernel_size=4,stride=2,padding=1,bias = False)    #64

    def forward(self,z_val,c_val):
        z = torch.cat((z_val,c_val),dim=1)   
        #print(z_val.shape,c_val.shape,z.shape)         
        x_f1 = self.dropout1(F.relu(self.fc1(z)))
        x_f2 = self.dropout2(F.relu(self.fc2(x_f1)))
        x_f3 = F.relu(self.fc3(x_f2))
        x_image = x_f3.reshape(-1,64,8,8)
        #print(x_image.shape)
        x_c1 =  F.relu((self.dconv1(x_image)))
        #print(x_c1.shape)
        x_c2 =  F.relu((self.dconv2(x_c1)))
        #print(x_c2.shape)
        out = self.dconv3(x_c2)
        return out


class CVAE(nn.Module):
    def __init__(self,d_prob=0.5,z_dim=10,c_dim=1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.encoder = encoder(d_prob,z_dim)
        self.decoder = decoder(d_prob,z_dim)
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
    
    def negative_elbo_bound(self, x,c_val): 
        m,v = self.encoder(x,c_val)
        z = sample_gaussian(m,v)
        logits = self.decoder(z,c_val)
        #print(m.shape,v.shape,z.shape,logits.shape)
        rec_tol = log_bernoulli_with_logits(x,logits)
        kl_tot = kl_normal(m,v,self.z_prior[0].expand(m.size()),self.z_prior[1].expand(v.size()))
        rec_loss = -1.0*torch.mean(rec_tol)
        kl_loss = torch.mean(kl_tot)
        nelbo = rec_loss + kl_loss
        return nelbo,rec_loss,kl_loss

    def sample_sigmoid(self,batch,c_val):
        z = sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))
        logits = self.decoder(z,c_val)
        return torch.sigmoid(logits)
    
    def sample_x(self,batch,c_val):
        log_prob = self.sample_sigmoid(batch,c_val)
        return torch.bernoulli(log_prob)


#helper functions
def sample_gaussian(m, v):
    epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size()))
    z = m + torch.sqrt(v) * epsilon
    return z

def gaussian_parameters(h, dim=-1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def log_bernoulli_with_logits(x, logits):
    log_prob = -bce(input=logits, target=x).sum(-1)
    return log_prob
