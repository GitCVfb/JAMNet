import os
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.autograd import Variable

from package_core.net_basics import *
from utils import conv1x1, actFunc, ModulatedDeformLayer
from utils_tf import feature_add_position
from transformer import FeatureTransformer


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
    return grid

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )
    
class ImageEncoder(nn.Module):
    def __init__(self, in_chs, init_chs, num_resblock=1):
        super(ImageEncoder, self).__init__()
        self.conv0 = conv2d(
            in_planes=in_chs,
            out_planes=init_chs[0],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=7,
            stride=1
        )
        self.resblocks0 = Cascade_resnet_blocks(in_planes=init_chs[0], n_blocks=num_resblock)
        self.conv1 = conv2d(
            in_planes=init_chs[0],
            out_planes=init_chs[1],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks1 = Cascade_resnet_blocks(in_planes=init_chs[1], n_blocks=num_resblock)
        self.conv2 = conv2d(
            in_planes=init_chs[1],
            out_planes=init_chs[2],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks2 = Cascade_resnet_blocks(in_planes=init_chs[2], n_blocks=num_resblock)
        self.conv3 = conv2d(
            in_planes=init_chs[2],
            out_planes=init_chs[3],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks3 = Cascade_resnet_blocks(in_planes=init_chs[3], n_blocks=num_resblock)
        self.conv4 = conv2d(
            in_planes=init_chs[3],
            out_planes=init_chs[4],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks4 = Cascade_resnet_blocks(in_planes=init_chs[4], n_blocks=num_resblock)

    def forward(self, x):
        x0 = self.resblocks0(self.conv0(x))
        x1 = self.resblocks1(self.conv1(x0))
        x2 = self.resblocks2(self.conv2(x1))
        x3 = self.resblocks3(self.conv3(x2))
        x4 = self.resblocks4(self.conv4(x3))

        return x4, x3, x2, x1, x0

class GSA(nn.Module):
    def __init__(self, n_feats):
        super(GSA, self).__init__()
        activation = 'prelu'
        self.F_f = nn.Sequential(
            nn.Linear(2 * n_feats, 4 * n_feats),
            actFunc(activation),
            nn.Linear(4 * n_feats, 2 * n_feats),
            nn.Sigmoid()
        )
        # condense layer
        self.condense = conv1x1(2 * n_feats, n_feats)
        self.act = actFunc(activation)

    def forward(self, cor):
        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        out = self.act(self.condense(w * cor))

        return out

class FusionModule(nn.Module):
    def __init__(self, n_feats):
        super(FusionModule, self).__init__()
        self.attention = GSA(n_feats)
        self.deform = ModulatedDeformLayer(n_feats, n_feats, deformable_groups=8)

    def forward(self, x):
        x = self.attention(x)
        return self.deform(x, x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :])
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :])
        out = self.prelu(x + self.conv5(out))
        return out

def predict_flow(in_channels):
    return nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, bias=True)

def predict_img(in_channels):
    return nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

class JAMDecorder_top_level(nn.Module):#top level
    def __init__(self, in_chs, hidd_chs):
        super(JAMDecorder_top_level, self).__init__()
        self.hidd_chs = hidd_chs

        self.convrelu = convrelu(in_chs*2, in_chs*2)

        self.resblocks = Cascade_resnet_blocks(in_planes=in_chs*2, n_blocks=3)
        
        self.predict_flow = predict_flow(in_chs*2)
        self.predict_img = predict_img(in_chs*2)

    def forward(self, c_0, c_1):
        x0 = torch.cat([c_0, c_1], dim=1)

        x0 = self.convrelu(x0)
        x_hidd = self.resblocks(x0)

        flow_curr = self.predict_flow(x_hidd)
        flow_0, flow_1 = flow_curr[:,:2], flow_curr[:,2:]
        img_pred = self.predict_img(x_hidd)

        return img_pred, x_hidd, flow_0, flow_1

class JAMDecorder_bottom_level(nn.Module):#second and third levels
    def __init__(self, in_chs_last, in_chs, hidd_chs):
        super(JAMDecorder_bottom_level, self).__init__()
        self.hidd_chs = hidd_chs

        self.deconv_flow = deconv(4, 4, kernel_size=4, stride=2, padding=1)
        self.deconv_hidden = deconv(2*in_chs_last, self.hidd_chs, kernel_size=4, stride=2, padding=1)

        self.convrelu = convrelu(in_chs*2+4+3+self.hidd_chs, in_chs*2)
        self.resblocks = Cascade_resnet_blocks(in_planes=in_chs*2, n_blocks=3)
        
        self.predict_img = predict_img(in_chs*2)

    def add_temporal_offset(self, F_pred, H, W):
        grid_rows = generate_2D_grid(H, W)[1]
        t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
    
        gamma = 1.0
        tau = gamma*(t_flow_offset-H//2)/H + 0.0001

        F_t_0 = (1.0-tau) * F_pred[:,0:2]
        F_t_1 = (-tau) * F_pred[:,2:4]
        return F_t_0, F_t_1
    
    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output
        
    def forward(self, img_last, hidden_last, flow_0_last, flow_1_last, c_0, c_1):
        upflow = self.deconv_flow(torch.cat([flow_0_last, flow_1_last], dim=1))
        uphidden = self.deconv_hidden(hidden_last)
        upimg = F.interpolate(img_last, scale_factor=2.0, mode='bilinear')
        
        upflow_0, upflow_1 = self.add_temporal_offset(upflow, upflow.size(2), upflow.size(3)) # Multiply time-offset
        
        warp_0 = self.warp(c_0, upflow_0)
        warp_1 = self.warp(c_1, upflow_1)
        
        x0 = torch.cat([warp_1, warp_0], dim=1)
        x0 = torch.cat([upimg, x0, uphidden, upflow_0, upflow_1], dim=1)
        x0 = self.convrelu(x0)
        x_hidd = self.resblocks(x0)

        img_pred = self.predict_img(x_hidd)

        return img_pred, x_hidd, upflow_0, upflow_1, upflow_0, upflow_1

class JAMDecorder(nn.Module):#second and third levels
    def __init__(self, in_chs_last, in_chs, hidd_chs):
        super(JAMDecorder, self).__init__()
        self.hidd_chs = hidd_chs

        self.deconv_flow = deconv(4, 4, kernel_size=4, stride=2, padding=1)
        self.deconv_hidden = deconv(2*in_chs_last, self.hidd_chs, kernel_size=4, stride=2, padding=1)

        self.convrelu = convrelu(in_chs*2+4+3+self.hidd_chs, in_chs*2)
        self.resblocks = Cascade_resnet_blocks(in_planes=in_chs*2, n_blocks=3)
        
        self.predict_flow = predict_flow(in_chs*2)
        self.predict_img = predict_img(in_chs*2)

    def add_temporal_offset(self, F_pred, H, W):
        grid_rows = generate_2D_grid(H, W)[1]
        t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
    
        gamma = 1.0
        tau = gamma*(t_flow_offset-H//2)/H + 0.0001

        F_t_0 = (1.0-tau) * F_pred[:,0:2]
        F_t_1 = (-tau) * F_pred[:,2:4]
        
        return F_t_0, F_t_1
        
    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output
        
    def forward(self, img_last, hidden_last, flow_0_last, flow_1_last, c_0, c_1):
        upflow = self.deconv_flow(torch.cat([flow_0_last, flow_1_last], dim=1))
        uphidden = self.deconv_hidden(hidden_last)
        upimg = F.interpolate(img_last, scale_factor=2.0, mode='bilinear')

        upflow_0, upflow_1 = self.add_temporal_offset(upflow, upflow.size(2), upflow.size(3)) # Multiply time-offset
        
        warp_0 = self.warp(c_0, upflow_0)
        warp_1 = self.warp(c_1, upflow_1)
        
        x0 = torch.cat([warp_1, warp_0], dim=1)
        x0 = torch.cat([upimg, x0, uphidden, upflow_0, upflow_1], dim=1)
        x0 = self.convrelu(x0)
        x_hidd = self.resblocks(x0)

        flow_curr = self.predict_flow(x_hidd) + upflow #flow residual link
        flow_0, flow_1 = flow_curr[:,:2], flow_curr[:,2:]
        img_pred = self.predict_img(x_hidd)

        return img_pred, x_hidd, flow_0, flow_1, upflow_0, upflow_1

class JAMNet(nn.Module):
    def __init__(self, hidd_chs=16):
        super(JAMNet, self).__init__()

        self.init_chs = [16, 28, 40, 64, 96]
        self.hidd_chs = hidd_chs
        self.attn_num_splits = 1
        
        self.encoder = ImageEncoder(in_chs=3, init_chs=self.init_chs)

        self.transformer = FeatureTransformer(num_layers=1,#
                                              d_model=self.init_chs[-1],
                                              nhead=1,
                                              attention_type='swin',
                                              ffn_dim_expansion=2,
                                              )

        self.decorder_5nd = JAMDecorder_top_level(self.init_chs[-1], self.hidd_chs)
        self.decorder_4nd = JAMDecorder(self.init_chs[-1], self.init_chs[-2], self.hidd_chs)
        self.decorder_3rd = JAMDecorder(self.init_chs[-2], self.init_chs[-3], self.hidd_chs)
        self.decorder_2nd = JAMDecorder(self.init_chs[-3], self.init_chs[-4], self.hidd_chs)
        self.decorder_1st = JAMDecorder_bottom_level(self.init_chs[-4], 3, self.hidd_chs)

    def forward(self, x):
        concat = torch.cat((x[:, :3], x[:, 3:]), dim=0)  #[2B, C, H, W]
        b,_,h,w=x.size()
        feature_4, feature_3, feature_2, feature_1, feature_0 = self.encoder(concat)
        c0_4, c0_3, c0_2, c0_1, c0_0 = feature_4[:b], feature_3[:b], feature_2[:b], feature_1[:b], feature_0[:b]
        c1_4, c1_3, c1_2, c1_1, c1_0 = feature_4[b:], feature_3[b:], feature_2[b:], feature_1[b:], feature_0[b:]
        
        # add position to features
        c0_4, c1_4 = feature_add_position(c0_4, c1_4, self.attn_num_splits, self.init_chs[-1])

        # Transformer
        c0_4, c1_4 = self.transformer(c0_4, c1_4, attn_num_splits=self.attn_num_splits)
        
        img_pred_4, x_hidd_4, flow_0_4, flow_1_4 = self.decorder_5nd(c0_4, c1_4)
        img_pred_3, x_hidd_3, flow_0_3, flow_1_3, upflow_0_3, upflow_1_3 = self.decorder_4nd(img_pred_4, x_hidd_4, flow_0_4, flow_1_4, c0_3, c1_3)
        img_pred_2, x_hidd_2, flow_0_2, flow_1_2, upflow_0_2, upflow_1_2 = self.decorder_3rd(img_pred_3, x_hidd_3, flow_0_3, flow_1_3, c0_2, c1_2)
        img_pred_1, x_hidd_1, flow_0_1, flow_1_1, upflow_0_1, upflow_1_1 = self.decorder_2nd(img_pred_2, x_hidd_2, flow_0_2, flow_1_2, c0_1, c1_1)
        img_pred_0, _,               _,        _, upflow_0_0, upflow_1_0 = self.decorder_1st(img_pred_1, x_hidd_1, flow_0_1, flow_1_1, x[:, :3], x[:, 3:])
        
        return img_pred_0,\
             [img_pred_0, upflow_0_0, upflow_1_0],\
             [img_pred_1, upflow_0_1, upflow_1_1],\
             [img_pred_2, upflow_0_2, upflow_1_2],\
             [img_pred_3, upflow_0_3, upflow_1_3],\
             [img_pred_4]


