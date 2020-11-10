import argparse

def add_args(parser):
    parser.add_argument('--buff_size',type=int,default=5000,
                        help='maximum size of the replay buffer')
    parser.add_argument('--n_trj',type=int,default=2500,
                        help='intial buffer quantity before training')
    parser.add_argument('--seq_len',type=int,default=5,
                        help='lenth of the RL episode')
    parser.add_argument('--num_episode',type=int,default=3000,
                        help='total number of training episode')
    parser.add_argument('--train',action='store_false',
                        help='whether to train or test the model')
    parser.add_argument('--reward_model',type=str,default='model/kl_strain.pt',
                        help='reard model to predict max strain')
    parser.add_argument('--model_name',type=str,default='kirigami_4',
                        help='default name of the model')
    parser.add_argument('--resume',action='store_true',
                        help='resume training from previous simulation')
    parser.add_argument('--resume_buffer',action='store_true',help='resume previous buffer')
    #structure into
    parser.add_argument('--Lx',type=float,default=200.0,
                        help='Lx length of kirigami binary map')
    parser.add_argument('--Ly',type=float,default=200.0,
                        help='Ly length of kirigami binary map')
    parser.add_argument('--ngrids',type=int,default=64,
                        help='No. of grids in x and y direction')
    parser.add_argument('--ny',type=int,default=4, help='total number of cuts')
    parser.add_argument('--l1',type=float,default=60.0, help='cut length 1')
    parser.add_argument('--l2',type=float,default=100.0, help='cut length 2')
    parser.add_argument('--l3',type=float,default=140.0, help='cut length 3')
    #environment parameters
    parser.add_argument('--threshold',type=float,default=15.0,
                        help='minimum strain for DQN to predict')
    parser.add_argument('--reward_scale',type=float,default=5.0,
                        help='factor by which strain is divided to compute reward')
    #DQN parameters
    parser.add_argument('--batch_size',type=int,default=64,
                        help='batch size for training RL agent')
    parser.add_argument('--gamma',type=float,default=0.70,
                        help='minimum strain for DQN to predict')
    parser.add_argument('--esp_start',type=float,default=0.60,
                        help='start value of epsilon')
    parser.add_argument('--esp_end',type=float,default=0.10,
                        help='end value of epsilon')
    parser.add_argument('--esp_decay',type=int,default=100,
                        help='reduce eps every n_th step')
    parser.add_argument('--targer_update',type=int,default=10,
                        help='update DQN traget')
    parser.add_argument('--learning_rate',type=float,default=0.0005,
                        help='learning_rate for DQN')
    #CVAE Parameters
    parser.add_argument('--z_dim',type=int,default=10,help='latent dimension')
    parser.add_argument('--ncvae_train',type=int,default=6400,help='total training examples used in CVAE')
    parser.add_argument('--dataset',type=str,default='dataset/4cut_allstate.npy',help='entire dataset of n-cut')
    parser.add_argument('--dataset_loc',type=str,default='dataset/4cut_location.npy',help='entire dataset of n-cut location')
    parser.add_argument('--num_epoch',type=int,default=300,
                        help='total number of training epochs')
    #Active Learning
    parser.add_argument('--nsearch',type=int,default=2,help='number of active learning search')

    return parser
    
def get_parser():
    parser = argparse.ArgumentParser(description='Parameters for training DQN model')
    parser = add_args(parser)
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args
    

