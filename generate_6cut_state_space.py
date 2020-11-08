from model.environment import generate_state
import numpy as np 
cut_generator_6 = generate_state(Lx=200.0,Ly=200.0,ngrids=64,ny=6,lengths=[50.0,100.0,150.0])

X_6cut,cut_6loc = cut_generator_6.generate_all_states_6cut(tot_states=None)
np.save('dataset/6cut_allstate',X_6cut)
np.save('dataset/6cut_location',cut_6loc)
print(X_6cut.shape,cut_6loc.shape)