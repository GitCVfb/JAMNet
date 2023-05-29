import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil
import imageio
import torch.nn.functional as F

import cv2
import flow_viz

from package_core.generic_train_test import *
from dataloader import *
from frame_utils import *
from model_JAM import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=True, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=False)
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=640, help='cropped image size width')

parser.add_argument('--model_label', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
 
opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelJAM(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Demo(Generic_train_test):
    def test(self):
        with torch.no_grad():
            seq_lists = os.listdir(self.opts.data_dir)
            for seq in seq_lists:
                im_rs0_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_0.png')
                im_rs1_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_1.png')

                im_rs0 = torch.from_numpy(io.imread(im_rs0_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()
                im_rs1 = torch.from_numpy(io.imread(im_rs1_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()

                im_rs = torch.cat([im_rs0,im_rs1], dim=1).float()/255.
                    
                _input = [im_rs, None, None, None]
                B,C,H,W = im_rs.size()
                
                self.model.set_input(_input)
                pred_gs_m_final, predFM_m_0, predFM_m_1, predFM_m_2 = self.model.forward()
                
                # save results
                im_gs_0_f = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_0_f.png'))
                im_gs_0_m = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_0_m.png'))
                im_gs_1_f = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1_f.png'))
                im_gs_1_m = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1_m.png'))
                im_rs_0 = io.imread(im_rs0_path)
                im_rs_1 = io.imread(im_rs1_path)

                im_gs_0_f = im_gs_0_f[:].copy()
                im_gs_0_m = im_gs_0_m[:].copy()
                im_gs_1_f = im_gs_1_f[:].copy()
                im_gs_1_m = im_gs_1_m[:].copy()
                im_rs_0 = im_rs_0[:].copy()
                im_rs_1 = im_rs_1[:].copy()

                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0.png'), im_rs_0)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs_1)
                
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_pred_m.png'), (pred_gs_m_final.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))

                print('saved', self.opts.results_dir, seq+'_pred_gs_img.png')
                
                im_gs_1_m_new=torch.from_numpy(im_gs_1_m[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_gt_m.png'), (im_gs_1_m_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                
                
    def warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()

        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        return output * mask              
    
    
    def compute_bmf_focus(self, flow_t_0, flow_t_1):
        nj_bmf_focus = torch.sum(torch.mul(torch.nn.functional.normalize(flow_t_0, dim=1), torch.nn.functional.normalize(flow_t_1, dim=1)), dim=1).unsqueeze(0)#Cos of the Angle
        
        wj_bmf_focus = torch.sqrt(1.0-(nj_bmf_focus * nj_bmf_focus))#Sin of the Angle
        
        wj_bmf_focus[ torch.where(wj_bmf_focus != wj_bmf_focus) ] = 0
        
        return wj_bmf_focus[0]
        
                        
Demo(model, opts, None, None).test()


