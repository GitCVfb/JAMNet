import os
import json
import torch
import random
import argparse
import numpy as np

from tensorboardX import SummaryWriter
from torchvision import transforms

from package_core.generic_train_test import *
from package_core.metrics import *
from dataloader import *
from model_JAM import *

torch.manual_seed(0)
##===================================================##
##********** Configure training settings ************##
##===================================================##

parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=8, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=False, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=256, help='cropped image size width')
parser.add_argument('--is_training', type=bool, default=True, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
parser.add_argument('--lr_step', type=int, default=260, help='lr decay rate')#260, 10 => 250 each lr_step epoch decay 0.5 times
parser.add_argument('--lr_start_epoch_decay', type=int, default=1000, help='epoch to start lr decay')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=True)

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--max_epochs', type=int, default=601)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--save_begin', type=int, default=0)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--model_label', type=str, default='pre', help='label used to load pre-trained model')

parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--dataset_root_dir', type=str, required=True, help='absolute path for training dataset')
parser.add_argument('--log_dir', type=str, required=True, help='directory used to store trained networks')

parser.add_argument('--load_1st_GS', type=bool, default=False)#Load two first scanline GS images?
parser.add_argument('--load_gt_flow', type=bool, default=False)#Load RS -> middle-row GS

parser.add_argument('--lamda_L1', type=float, default=10)
parser.add_argument('--lamda_perceptual', type=float, default=1)
parser.add_argument('--lamda_flow_smoothness', type=float, default=0.1)

opts=parser.parse_args()

opts.log_dir = opts.log_dir+opts.dataset_type+'_t1'\
                '_Percep'+str(opts.lamda_perceptual)+\
                '_L1'+str(opts.lamda_L1)+\
                '_FlowTV'+str(opts.lamda_flow_smoothness)

if not os.path.exists(opts.log_dir):
    #print(opts.log_dir)
    os.makedirs(opts.log_dir)

config_save_path = os.path.join(opts.log_dir, 'opts.txt')
with open(config_save_path, 'w') as f:
    json.dump(opts.__dict__, f, indent=2)

##===================================================##
##*************** Create dataloader *****************##
##===================================================##
dataloader = Create_dataloader(opts)

# update opts
orig_batch_sz = opts.batch_sz
opts.dataset_root_dir = opts.dataset_root_dir.replace('/train', '/val')
opts.batch_sz = 1

dataloader_val = Create_dataloader(opts)

# recover 
opts.dataset_root_dir = opts.dataset_root_dir.replace('/val', '/train')
opts.batch_sz = orig_batch_sz

##===================================================##
##*************** Create datalogger *****************##
##===================================================##
logger = SummaryWriter(opts.log_dir)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelJAM(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Train(Generic_train_test):
    def augment_data(self, _input):
        im_rs, im_gs, flow, im_gs_f = _input
        B,C,H,W=im_rs.size() 
        
        # do random crop
        cH=random.randint(0, H-self.opts.crop_sz_H)
        cW=random.randint(0, W-self.opts.crop_sz_W)
        
        im_rs = im_rs[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()
        im_gs = im_gs[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()
        if im_gs_f is not None:
            im_gs_f = im_gs_f[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()
        
        flow = None
        if flow is not None:
            flow = flow[:,:,cH:cH+self.opts.crop_sz_H, cW:cW+self.opts.crop_sz_W].clone()#Donot process
        
        # Data augmentation strategy 1
        # Add horizontally flip
        im_rs_=im_rs[0,:,:,:].unsqueeze(0)
        im_gs_=im_gs[0,:,:,:].unsqueeze(0)  
        if im_gs_f is not None:
            im_gs_f_=im_gs_f[0,:,:,:].unsqueeze(0)  
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()])
        for i in range(1,B):
            if torch.rand(1, 1) > 0.5:
                im_rs_1 = im_rs[i,0:3,:,:].squeeze(0)
                im_rs_2 = trans(im_rs_1).unsqueeze(0)
                im_rs_1 = im_rs[i,3:6,:,:].squeeze(0)
                im_rs_3 = trans(im_rs_1).unsqueeze(0)
                im_rs_4 = torch.cat([im_rs_2,im_rs_3],1)
                im_rs_ = torch.cat([im_rs_,im_rs_4],0)
    
                im_gs_1 = im_gs[i,0:3,:,:].squeeze(0)
                im_gs_2 = trans(im_gs_1).unsqueeze(0)
                im_gs_1 = im_gs[i,3:6,:,:].squeeze(0)
                im_gs_3 = trans(im_gs_1).unsqueeze(0)
                im_gs_4 = torch.cat([im_gs_2,im_gs_3],1)
                im_gs_ = torch.cat([im_gs_,im_gs_4],0)
                
                if im_gs_f is not None:
                    im_gs_1 = im_gs_f[i,0:3,:,:].squeeze(0)
                    im_gs_2 = trans(im_gs_1).unsqueeze(0)
                    im_gs_1 = im_gs_f[i,3:6,:,:].squeeze(0)
                    im_gs_3 = trans(im_gs_1).unsqueeze(0)
                    im_gs_4 = torch.cat([im_gs_2,im_gs_3],1)
                    im_gs_f_ = torch.cat([im_gs_f_,im_gs_4],0)
            else:
                im_rs_ = torch.cat([im_rs_,im_rs[i,:,:,:].unsqueeze(0)],0)
                im_gs_ = torch.cat([im_gs_,im_gs[i,:,:,:].unsqueeze(0)],0)
                if im_gs_f is not None:
                    im_gs_f_ = torch.cat([im_gs_f_,im_gs_f[i,:,:,:].unsqueeze(0)],0)            
        im_rs=im_rs_
        im_gs=im_gs_
        if im_gs_f is not None:
            im_gs_f=im_gs_f_
        
        # Data augmentation strategy 2: vertical flip + inverse order
        im_rs_=im_rs[0,:,:,:].unsqueeze(0)
        im_gs_=im_gs[0,:,:,:].unsqueeze(0)  
        if im_gs_f is not None:
            im_gs_f_=im_gs_f[0,:,:,:].unsqueeze(0)  
        trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()])
        for i in range(1,B):
            if torch.rand(1, 1) > 0.5:
                im_rs_1 = im_rs[i,3:6,:,:].squeeze(0)
                im_rs_2 = trans(im_rs_1).unsqueeze(0)
                im_rs_1 = im_rs[i,0:3,:,:].squeeze(0)
                im_rs_3 = trans(im_rs_1).unsqueeze(0)
                im_rs_4 = torch.cat([im_rs_2,im_rs_3],1)
                im_rs_ = torch.cat([im_rs_,im_rs_4],0)
    
                im_gs_1 = im_gs[i,3:6,:,:].squeeze(0)
                im_gs_2 = trans(im_gs_1).unsqueeze(0)
                im_gs_1 = im_gs[i,0:3,:,:].squeeze(0)
                im_gs_3 = trans(im_gs_1).unsqueeze(0)
                im_gs_4 = torch.cat([im_gs_2,im_gs_3],1)
                im_gs_ = torch.cat([im_gs_,im_gs_4],0)
                
                if im_gs_f is not None:
                    im_gs_1 = im_gs_f[i,0:3,:,:].squeeze(0)
                    im_gs_2 = trans(im_gs_1).unsqueeze(0)
                    im_gs_1 = im_gs_f[i,3:6,:,:].squeeze(0)
                    im_gs_3 = trans(im_gs_1).unsqueeze(0)
                    im_gs_4 = torch.cat([im_gs_2,im_gs_3],1)
                    im_gs_f_ = torch.cat([im_gs_f_,im_gs_4],0)
            else:
                im_rs_ = torch.cat([im_rs_,im_rs[i,:,:,:].unsqueeze(0)],0)
                im_gs_ = torch.cat([im_gs_,im_gs[i,:,:,:].unsqueeze(0)],0)
                if im_gs_f is not None:
                    im_gs_f_ = torch.cat([im_gs_f_,im_gs_f[i,:,:,:].unsqueeze(0)],0)            
        im_rs=im_rs_
        im_gs=im_gs_
        if im_gs_f is not None:
            im_gs_f=im_gs_f_
        
        
        return [im_rs, im_gs, flow, im_gs_f]

    def decode_input(self, data):
        im_rs=data['I_rs']
        im_gs=data['I_gs']

        im_gs_f=None
        if self.opts.load_1st_GS:
            im_gs_f=data['I_gs_f']
        flow=None
        if self.opts.load_gt_flow:
            flow=data['flow_m']
            
        _input = [im_rs, im_gs, flow, im_gs_f]
        return self.augment_data(_input)

    def validation(self, epoch):
        psnr_sum = 0.
        ssim_sum = 0.
        '''
        for i, data in enumerate(self.dataloader_val):
            # decode input
            im_rs=data['I_rs'].cuda()
            im_gs_m=data['I_gs'].cuda()
            flow=data['flow'][:,:2,:,:].clone().cuda()
            
            _input = [im_rs, im_gs_m, None, None]
            self.model.set_input(_input)

            with torch.no_grad():
                pred_im, _, _, _ = self.model.forward()

            # compute metrics 
            psnr_sum += PSNR(pred_im[1], im_gs_m[:,-3:,:,:])
            ssim_sum += SSIM(pred_im[1], im_gs_m[:,-3:,:,:])

        psnr_avg = psnr_sum/len(self.dataloader_val)
        ssim_avg = ssim_sum/len(self.dataloader_val)

        self.logger.add_scalar('PSNR_val', psnr_avg, epoch)
        self.logger.add_scalar('SSIM_val', ssim_avg, epoch)
        '''

Train(model, opts, dataloader, logger, dataloader_val).train()


