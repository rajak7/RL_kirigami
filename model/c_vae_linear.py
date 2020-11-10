import torch
import torch.nn as nn
import torch.nn.functional as F

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

class encoder(nn.Module):
    def __init__(self,d_prob=0.5,z_dim=10):
        super(encoder, self).__init__()
        self.input_dim = 4096  #64*64
        self.d_prob = d_prob
        self.z_dim = z_dim
        self.fc1=nn.Linear(self.input_dim+1,2048)
        self.dropout1 = nn.Dropout(p = self.d_prob)
        self.fc2=nn.Linear(2048,1024)
        self.dropout2 = nn.Dropout(p = self.d_prob)
        self.fc3=nn.Linear(1024,512)
        self.dropout3 = nn.Dropout(p = self.d_prob)
        self.fc4=nn.Linear(512,2*self.z_dim)
        self.act = F.relu

    def forward(self,X,Y):
        #print(x_flat.shape,Y.shape)
        input_XY = torch.cat((X,Y),dim=1)
        x_f1= (self.act(self.fc1(input_XY)))   #self.dropout1
        x_f2= (self.act(self.fc2(x_f1)))
        x_f3= (self.act(self.fc3(x_f2)))
        h= self.fc4(x_f3)
        m,v = gaussian_parameters(h,dim=1)
        return m,v

class decoder(nn.Module):
    def __init__(self,d_prob=0.5,z_dim=10,c_dim=1):
        super(decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.d_prob = d_prob
        self.output_dim = 4096  #64*64
        self.fc1=nn.Linear(self.z_dim+self.c_dim,512)
        self.dropout1 = nn.Dropout(p = self.d_prob)
        self.fc2=nn.Linear(512,1024)
        self.dropout2 = nn.Dropout(p = self.d_prob)
        self.fc3=nn.Linear(1024,2048)
        self.dropout3 = nn.Dropout(p = self.d_prob)
        self.fc4=nn.Linear(2048,self.output_dim)

    def forward(self,z_val,c_val):
        z = torch.cat((z_val,c_val),dim=1)   
        #print(z_val.shape,c_val.shape,z.shape)         
        x_f1 = (F.relu(self.fc1(z)))    #self.dropout1
        x_f2 = (F.relu(self.fc2(x_f1)))
        x_f3 = (F.relu(self.fc3(x_f2)))
        out =self.fc4(x_f3)
        return out


class CVAE(nn.Module):
    def __init__(self,d_prob=0.5,z_dim=10,c_dim=1):
        super().__init__()
        self.input_dim = 4096
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.encoder = encoder(d_prob,z_dim)
        self.decoder = decoder(d_prob,z_dim)
        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)
    
    def negative_elbo_bound(self, x,c_val): 
        x_flat = x.reshape(-1,self.input_dim)
        m,v = self.encoder(x_flat,c_val)
        z = sample_gaussian(m,v)
        logits = self.decoder(z,c_val)
        #print(m.shape,v.shape,z.shape,logits.shape)
        rec_tol = log_bernoulli_with_logits(x_flat,logits)
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
