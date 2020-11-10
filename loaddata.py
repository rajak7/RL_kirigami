import numpy as np
import math
from model.environment import generate_state
from torch.utils import data

def read_input(file1,file2,ncopy=1):
    train_XX=np.load(file1)
    train_YY=list()
    with open(file2,'r') as in_file:
        val=in_file.readline()
        for val in in_file:
            val=val.strip().split()
            x_initial=float(val[1])
            x_finish=float(val[2])
            strain = 100.0*(x_finish-x_initial)/x_initial
            for n in range(ncopy):
                train_YY.append(strain)
    train_YY=np.asarray(train_YY)
    return train_XX,train_YY

def read_stress_strain(nfiles,dir_name):
    stress_tot  =[]
    strain_tot = []
    for ii in range(nfiles):
        with open(dir_name+str(ii+1)+'.txt','r') as in_file:
            val = in_file.readline()
            stress = [0.0]
            strain = [0.0]
            for val in in_file:
                val=val.strip().split()
                strain.append(float(val[1]))
                stress.append(float(val[2]))
            strain[0]=float(val[0])
            stress_tot.append(stress)
            strain_tot.append(strain)
    stress_tot = np.asarray(stress_tot)
    strain_tot = np.asarray(strain_tot)
    return stress_tot,strain_tot

#read input cut location from file from file and create  a binary structure map
def create_krigami_binary_map(filename,s_info):
    cut_generator = generate_state(s_info['Lx'],s_info['Ly'],s_info['ngrids'],s_info['ny'],s_info['lengths'])
    action_id = dict()
    for key in cut_generator.action:
        val = cut_generator.action[key]
        cut_loc = str('('+str(val[0]/s_info['Lx'])+' '+str(val[1])+')')
        action_id[cut_loc] = key

    with open(filename,'r') as infile:
        val = infile.readline()
        cut_id_all = []
        cut_action_all = []
        structure_all = []
        for count,val in enumerate(infile):
            val = val.strip()
            p = val.split(';')
            cut_id = int(p[0])
            cut1 = p[1].split(',')
            cut2 = p[2].split(',')
            #cut 1 id 
            cut1_action = []
            for val in cut1:
                if val not in action_id:
                    print("undefined cut id",val)
                    print("undefined cut id",cut_id,cut1)
                cut1_action.append(action_id[val])
            cut2_action = []
            for val in cut2:
                if val not in action_id:
                    print("undefined cut id",val)
                    print("undefined cut id",cut_id,cut2)
                cut2_action.append(action_id[val])
            #add the data
            cut_id_all.append(cut_id)
            X1 = cut_generator.create_n_cut_structure(cut1_action) 
            X2 = cut_generator.create_n_cut_structure(cut2_action) 
            cut_action_all.append(cut1_action)
            cut_action_all.append(cut2_action)
            structure_all.append(X1[0])
            structure_all.append(X2[0])
    print("total number of structure: ",count)
    return cut_id_all,cut_action_all,structure_all

#create queantity of each pahse in the structure 
class Kirigami_Dataset(data.Dataset):
    def __init__(self,structure,strain):
        n_data = structure.size(0)   
        self.structure = structure
        self.strain =strain

    def __len__(self):
        return self.structure.size(0)

    def __getitem__(self, index):
        X_ = self.structure[index]  
        C_ = self.strain[index]     
        return {'structure':X_,'strain':C_}
