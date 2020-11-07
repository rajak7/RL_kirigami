import sys
import numpy as np

n_files=int(sys.argv[1])
#input_dir="Data/"
#input_dir="~/Desktop/viz/kirigami/kirigami/"
input_dir=sys.argv[2]
f_name_start="data.MoS2_kirigami_"
shift=20.0
ngirds=64

print("Total number of files: ",n_files)


def readinput_kirigami(input_dir,f_name_start,n_files):
    s_val=input_dir+f_name_start
    train_XX=np.zeros((n_files,ngirds,ngirds),dtype=float)
    for ii in range(n_files):
        f_index=ii+1
        f_name=s_val+str(f_index)
        with open(f_name,'r') as in_file:
            val=in_file.readline()
            val=in_file.readline()
            natoms=in_file.readline().strip().split()
            ntype=in_file.readline().strip().split()
            val=in_file.readline()
            val=in_file.readline().strip().split()
            Lx_min=float(val[0])
            Lx_max=float(val[1])
            val=in_file.readline().strip().split()
            Ly_min=float(val[0])
            Ly_max=float(val[1])
            val=in_file.readline().strip().split()
            Lz_min=float(val[0])
            Lz_max=float(val[1])
            x_bound=[0.0,Lx_max-shift]
            y_bound=[0.0,Ly_max]
            del_x=x_bound[1]/float(ngirds)
            del_y=y_bound[1]/float(ngirds)
            #print(Lx_min,Lx_max,Ly_min,Ly_max,Lz_min,Lz_max)
            #print(x_bound,y_bound,del_x,del_y)
            val=in_file.readline()
            val=in_file.readline()
            val=in_file.readline()
            for count,val in enumerate(in_file):
                val=val.strip().split()
                itype=int(val[2])
                if itype <= 2:
                    x_cor=float(val[4])
                    y_cor=float(val[5])
                    x_grid=int(x_cor/del_x)
                    y_grid=int(y_cor/del_y)
                    if x_grid > ngirds-1 or y_grid > ngirds-1:
                        print(x_grid,y_grid,ngirds-1)
                        exit(1)
                    train_XX[ii,y_grid,x_grid]=1
        print("Total atoms: ",ii,count,natoms)
    return train_XX


#---------------------
train_XX=readinput_kirigami(input_dir,f_name_start,n_files)
np.save('kirigami_struct',train_XX)
