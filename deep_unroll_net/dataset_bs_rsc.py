from pickle import FALSE
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage import io
from frame_utils import *

class Dataset_bs_rsc(Dataset):
    def __init__(self, root_dir, seq_len=2):
        self.I_rs = []
        self.I_gs = []
        self.seq_len = seq_len
        
        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname != '00000.png':
                    continue
                    
                # read in seq of images            
                for i in range(50-seq_len+1):
                    if not os.path.exists(os.path.join(seq_path.replace('/RS','/'), 'RS/'+str(i).zfill(5)+'.png')):
                        continue

                    seq_Irs=[]
                    seq_Igs=[]

                    seq_Irs.append(os.path.join(seq_path.replace('/RS','/'), 'RS/'+str(i).zfill(5)+'.png'))
                    seq_Igs.append(os.path.join(seq_path.replace('/RS','/'), 'GS/'+str(i).zfill(5)+'.png'))
                    
                    #print(os.path.join(seq_path.replace('/RS','/'), 'RS/'+str(i).zfill(5)+'.png'))
                    #print(os.path.join(seq_path.replace('/RS','/'), 'GS/'+str(i).zfill(5)+'.png'))
                    #print('\n')
                    
                    for j in range(1,seq_len):
                        seq_Irs.append(os.path.join(seq_path.replace('/RS','/'), 'RS/'+str(i+j).zfill(5)+'.png'))
                        seq_Igs.append(os.path.join(seq_path.replace('/RS','/'), 'GS/'+str(i+j).zfill(5)+'.png'))

                    if not os.path.exists(seq_Irs[-1]):
                        break

                    self.I_rs.append(seq_Irs.copy())
                    self.I_gs.append(seq_Igs.copy())

    def __len__(self):
        return len(self.I_gs)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]

        temp = io.imread(path_rs[0])
        H,W,C=temp.shape
        if C>3:
            C=3

        I_rs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        I_gs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        
        for i in range(self.seq_len):
            I_rs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_rs[i]).transpose(2,0,1)).float()[:3]/255.
            I_gs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_gs[i]).transpose(2,0,1)).float()[:3]/255.

        return {'I_rs': I_rs,
                'I_gs': I_gs,
                'I_gs_f': I_gs,
                'path': path_rs}

