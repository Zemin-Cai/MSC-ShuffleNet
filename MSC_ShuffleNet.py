import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
from utils import *
__all__ = ['UNext']
from torch.nn.parameter import Parameter
import math
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
import numpy as np
import torch
from torch import nn
from math import sqrt
import cv2




class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(sa_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel, 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel , 1, 1))
        self.sbias = Parameter(torch.ones(1, channel , 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel , channel)

    @staticmethod
    def channel_shuffle(x, groups):
        b, h, w,c = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=x.permute(0, 3, 1, 2).contiguous()
        b, c, h, w = x.shape

        #x = x.reshape(b * self.groups, -1, h, w)
        #x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        '''
        xn = self.avg_pool(x)
        xn = self.cweight * xn + self.cbias
        xn = x * self.sigmoid(xn)
        '''
        xs = self.gn(x)
        xs = self.sweight * xs + self.sbias
        xs = x * self.sigmoid(xs)
        out = xs.reshape(b, -1, h, w)
        out = out.permute(0, 2, 3, 1).contiguous()
        # concatenate along channel axis

        #out = self.channel_shuffle(out, 2)
        return out

class ca_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel, 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel , 1, 1))
        self.sbias = Parameter(torch.ones(1, channel , 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel , channel)

    def forward(self, x):

        b, c, h, w = x.shape

        #x = x.reshape(b * self.groups, -1, h, w)
        #x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x)
        xn = self.cweight * xn + self.cbias
        xn = x * self.sigmoid(xn)


        out = xn.reshape(b, -1, h, w)

        # concatenate along channel axis

        #out = self.channel_shuffle(out, 2)
        return out


def channel_shuffle(x, groups):
    b, c, h, w = x.shape

    x = x.reshape(b, groups, -1, h, w)
    x = x.permute(0, 2, 1, 3, 4)

    # flatten
    x = x.reshape(b, -1, h, w)

    return x

def window_shift(shifted_x, window_size):

    temp1 = shifted_x[:, 1*window_size:2*window_size, :, :].clone()
    shifted_x[:, 1*window_size:2*window_size, :, :]=shifted_x[:, 2*window_size:3*window_size, :, :]
    shifted_x[:, 2*window_size:3*window_size, :, :]=temp1

    temp2 =shifted_x[:, :, 1*window_size:2*window_size,:].clone()
    shifted_x[:, :, 1*window_size:2*window_size,:] = shifted_x[:, :, 2*window_size:3*window_size,:]
    shifted_x[:, :, 2*window_size:3*window_size,:]=temp2

    return shifted_x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def dilated_shift(x_dilated, window_size):

    shift_x = torch.cat([x_dilated[:, 0:window_size:2, :, :], x_dilated[:, 1:window_size + 1:2, :, :]], dim=1)
    shift_x = torch.cat([shift_x[:, :, 0:window_size:2, :], shift_x[:, :, 1:window_size + 1:2, :]], dim=2)

    return shift_x

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def dilated_shift_reverse(shifted_x, window_size):
    # shifted_x_detach = shifted_x.permute(0, 3, 1, 2).detach().cpu().numpy()
    '''
    for i in range(0, window_size):
        temp = shifted_x[:, i, :, :].clone()
        shifted_x[:, i, :, :]=shifted_x[:, i+window_size, :, :]
        shifted_x[:, i + window_size, :, :]=temp
    for i in range(0, window_size):
        temp =shifted_x[:, :, i, :].clone()
        shifted_x[:, :, i, :] = shifted_x[:,:, i + window_size, :]
        shifted_x[:, :, i + window_size, :]=temp
    '''
    x_row = None
    for i in range(0, window_size):
        if x_row is None:
            x_row = torch.cat([shifted_x[:, i, :, :].unsqueeze(1), shifted_x[:, i + window_size, :, :].unsqueeze(1)], dim=1)
        else:
            x_row = torch.cat([x_row, shifted_x[:, i, :, :].unsqueeze(1), shifted_x[:, i + window_size, :, :].unsqueeze(1)], dim=1)
        # x = x_row
        # x_row_detach = x_row.permute(0, 3, 1, 2).detach().cpu().numpy()
    x_col = None
    for j in range(0, window_size):
        if x_col is None:
            x_col = torch.cat([x_row[:, :, j, :].unsqueeze(2), x_row[:, :, j + window_size, :].unsqueeze(2)], dim=2)
        else:
            x_col = torch.cat([x_col, x_row[:, :, j, :].unsqueeze(2), x_row[:, :, j + window_size, :].unsqueeze(2)], dim=2)
        # x_col_detach = x_col.permute(0, 3, 1, 2).detach().cpu().numpy()

    return x_col


class MSC_ShuffleNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes,  embed_dims=[128,160,256,320]
                 , drop_path_rate=0., norm_layer=nn.LayerNorm,window_size=8,dilated_window_size=16,
                 depths=[1, 1, 1], **kwargs):
        super().__init__()

        self.window_size1=window_size
        self.window_size2 = window_size//2
        self.window_size3 = window_size//4

        self.dilated_window_size1=dilated_window_size
        self.dilated_window_size2 = dilated_window_size//2
        self.dilated_window_size3 = dilated_window_size//4
        self.shift_size=4
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        self.relu4 = nn.LeakyReLU()
        self.relu5 = nn.LeakyReLU()


        self.lower_channel1=nn.Conv2d(96, 32, 1, stride=1, padding=0)
        self.lower_channel2 = nn.Conv2d(120, 40, 1, stride=1, padding=0)
        self.lower_channel3 = nn.Conv2d(192, 64, 1, stride=1, padding=0)

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv_downsampling1 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_downsampling2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.convs_32 = nn.ModuleList([])
        for i in range(3):
            self.convs_32.append(nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=3 + i*2, stride=1, padding=i+1,groups=2),

                nn.Conv2d(8, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=2, stride=2),
                nn.LeakyReLU()
            ))
        #self.encoder2 = nn.Conv2d(4, 8, 3, stride=1, padding=1,dilation=1)
        self.convs_128 = nn.ModuleList([])
        for i in range(3):
            self.convs_128.append(nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3 + i * 2, stride=1, padding=i + 1,groups=8),

                nn.Conv2d(32, 40, kernel_size=1),
                nn.BatchNorm2d(40),
                nn.Conv2d(40, 40, kernel_size=2, stride=2),
                nn.LeakyReLU()
            ))
        #self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.convs_160 = nn.ModuleList([])
        for i in range(3):
            self.convs_160.append(nn.Sequential(
                nn.Conv2d(40, 40, kernel_size=3 + i * 2, stride=1, padding=i + 1,groups=10),
                nn.Conv2d(40, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=2, stride=2),
                nn.LeakyReLU()
            ))

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)


        self.norm2 = norm_layer(embed_dims[0]//4)
        self.norm3 = norm_layer(embed_dims[1]//4)
        self.norm4 = norm_layer(embed_dims[2]//4)
        self.norm5 = norm_layer(embed_dims[3]//4)

        self.dnorm2 = norm_layer(32)
        self.dnorm3 = norm_layer(40)
        self.dnorm4 = norm_layer(256)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.sa1=sa_layer(3*embed_dims[0] // 4)
        self.sa2 = sa_layer(3*embed_dims[1] // 4)
        self.sa3 = sa_layer(3*embed_dims[2] // 4)
        self.ca1 = ca_layer(3*embed_dims[0] // 4)
        self.ca2 = ca_layer(3*embed_dims[1] // 4)
        self.ca3 = ca_layer(3*embed_dims[2] // 4)

        self.decoder_256 = nn.ModuleList([])
        for i in range(3):
            self.decoder_256.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3 + i * 2, stride=1, padding=i + 1,groups=16),
                nn.Conv2d(64, 40, kernel_size=1),
                nn.BatchNorm2d(40),
                nn.Upsample(scale_factor=2,mode ='bilinear'),
                nn.LeakyReLU()
            ))

        self.decoder_160 = nn.ModuleList([])
        for i in range(3):
            self.decoder_160.append(nn.Sequential(
                nn.Conv2d(40, 40, kernel_size=3 + i * 2, stride=1, padding=i + 1,groups=10),
                nn.Conv2d(40, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.LeakyReLU()
            ))
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(32, 16 , 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)


        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.sigmoid = nn.Sigmoid()

        self.final = nn.Conv2d(int(16), num_classes, kernel_size=1, stride=1, padding=0)

        #self.decoderf = nn.Conv2d(int(16), int(16), kernel_size=3, stride=1, padding=1)

        #self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = self.relu1(self.conv_downsampling1(self.ebn1(self.encoder1(x))))
        t1 = out
        out = self.relu2(self.conv_downsampling2(self.ebn2(self.encoder2(out))))
        t2 = out
        ### Stage 2

        xs = torch.chunk(out, self.shift_size, 1)
        for i,xin in enumerate(xs):
            if i == 0:
                xins = xin
            else:
                if i % 2 == 0:
                    if i==2:
                        xin_US=xins
                        xins = xin
                    else:
                        xin_US=torch.cat([xin_US, xins], dim=3)
                        xins=xin
                else:
                    xins = torch.cat([xin,xins],dim=2)
                    if i==3:
                        xin_US = torch.cat([xin_US, xins], dim=3)
        out=xin_US
        for j, conv in enumerate(self.convs_32):
            fea = conv(out)
            if j == 0:
                feas=fea
            else:
                feas=torch.cat([fea,feas], dim=1)
        #outs = self.lower_channel1(out)
        feas=channel_shuffle(feas,3)
        _, _, H1, W1 = feas.shape
        x_win = feas.permute(0, 2, 3, 1).contiguous()
        # x_win = window_shift(x_win, H1 // 4)
        x_win = dilated_shift(x_win, H1)
        #x_win = window_partition(x_win, H1 // 2)
        # _,H2,W2,_=x_win.shape
        # x_win = dilated_shift(x_win, self.dilated_window_size1)
        # x_win = window_partition(x_win, self.window_size1)

        sa_attns = self.sa1(x_win)
        #sa_attns = window_reverse(sa_attns, H1 // 2, H1, W1)

        # sa_attn=window_reverse(sa_attn,self.window_size1,H2,W2)
        sa_attns = dilated_shift_reverse(sa_attns, H1 // 2)
        # sa_attn=window_reverse(sa_attn,self.dilated_window_size1,H1,W1)

        # sa_attn = window_shift(sa_attn, H1 // 4)
        sa_attns = sa_attns.permute(0, 3, 1, 2).contiguous()
        sa_attns=self.ca1(sa_attns)
        sa_attns += feas
        sa_attns=self.lower_channel1(sa_attns)
        out=sa_attns

        out=self.relu3(out)
        B, C, H, W = out.shape
        out = out.reshape(B, -1, C)
        out = self.norm2(out)
        out = out.reshape(B, C, H, W)
        t3 = out


        for j, conv in enumerate(self.convs_128):
            fea = conv(out)
            if j == 0:
                feas=fea
            else:
                feas=torch.cat([fea,feas], dim=1)
            #outs = self.lower_channel1(out)
        feas = channel_shuffle(feas, 3)
        _, _, H1, W1 = feas.shape
        x_win = feas.permute(0, 2, 3, 1).contiguous()
        # x_win = window_shift(x_win, H1 // 4)
        x_win = dilated_shift(x_win, H1)
        #x_win = window_partition(x_win, H1 // 2)
        # _,H2,W2,_=x_win.shape
        # x_win = dilated_shift(x_win, self.dilated_window_size1)
        # x_win = window_partition(x_win, self.window_size1)

        sa_attns = self.sa2(x_win)
        #sa_attns = window_reverse(sa_attns, H1 // 2, H1, W1)

        # sa_attn=window_reverse(sa_attn,self.window_size1,H2,W2)
        sa_attns = dilated_shift_reverse(sa_attns, H1 // 2)
        # sa_attn=window_reverse(sa_attn,self.dilated_window_size1,H1,W1)

        # sa_attn = window_shift(sa_attn, H1 // 4)
        sa_attns = sa_attns.permute(0, 3, 1, 2).contiguous()
        sa_attns = self.ca2(sa_attns)
        sa_attns += feas
        sa_attns = self.lower_channel2(sa_attns)
        out = sa_attns
        out =self.relu4(out)
        B, C, H, W = out.shape
        out = out.reshape(B, -1, C)
        out = self.norm3(out)
        out = out.reshape(B, C, H, W)
        t4 = out

        for j, conv in enumerate(self.convs_160):
            fea = conv(out)
            if j == 0:
                feas = fea
            else:
                feas = torch.cat([fea, feas], dim=1)
                # outs = self.lower_channel1(out)
        feas = channel_shuffle(feas, 3)
        _, _, H1, W1 = feas.shape
        x_win = feas.permute(0, 2, 3, 1).contiguous()
        # x_win = window_shift(x_win, H1 // 4)
        x_win = dilated_shift(x_win, H1)
        #x_win = window_partition(x_win, H1 // 2)
        # _,H2,W2,_=x_win.shape
        # x_win = dilated_shift(x_win, self.dilated_window_size1)
        # x_win = window_partition(x_win, self.window_size1)

        sa_attns = self.sa3(x_win)
        #sa_attns = window_reverse(sa_attns, H1 // 2, H1, W1)

        # sa_attn=window_reverse(sa_attn,self.window_size1,H2,W2)
        sa_attns = dilated_shift_reverse(sa_attns, H1 // 2)
        # sa_attn=window_reverse(sa_attn,self.dilated_window_size1,H1,W1)

        # sa_attn = window_shift(sa_attn, H1 // 4)
        sa_attns = sa_attns.permute(0, 3, 1, 2).contiguous()
        sa_attns = self.ca3(sa_attns)
        sa_attns += feas
        sa_attns = self.lower_channel3(sa_attns)
        out = sa_attns
        out = self.relu5(out)
        B, C, H, W = out.shape
        out = out.reshape(B, -1, C)
        out = self.norm4(out)
        out = out.reshape(B, C, H, W)

        #out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        #xs = torch.chunk(out, self.shift_size, 1)
        #for i, xin in enumerate(xs):

        for j, conv in enumerate(self.decoder_256):
            fea = conv(out)
            if j == 0:
                feas = fea
            else:
                feas = torch.cat([fea, feas], dim=1)

            # outs = self.lower_channel1(out)
        feas = channel_shuffle(feas, 3)
        _, _, H1, W1 = feas.shape
        x_win = feas.permute(0, 2, 3, 1).contiguous()
        # x_win = window_shift(x_win, H1 // 4)
        x_win = dilated_shift(x_win, H1)
        #x_win = window_partition(x_win, H1 // 2)
        # _,H2,W2,_=x_win.shape
        # x_win = dilated_shift(x_win, self.dilated_window_size1)
        # x_win = window_partition(x_win, self.window_size1)

        sa_attns = self.sa2(x_win)
        #sa_attns = window_reverse(sa_attns, H1 // 2, H1, W1)

        # sa_attn=window_reverse(sa_attn,self.window_size1,H2,W2)
        sa_attns = dilated_shift_reverse(sa_attns, H1 // 2)
        # sa_attn=window_reverse(sa_attn,self.dilated_window_size1,H1,W1)

        # sa_attn = window_shift(sa_attn, H1 // 4)
        sa_attns = sa_attns.permute(0, 3, 1, 2).contiguous()
        sa_attns = self.ca2(sa_attns)
        sa_attns += feas
        sa_attns = self.lower_channel2(sa_attns)
        out = sa_attns
        out = self.relu4(out)
        B, C, H, W = out.shape
        out = out.reshape(B, -1, C)
        out = self.dnorm3(out)
        out = out.reshape(B, C, H, W)
        out = torch.add(out,t4)
        '''
        B, C, H, W = out.shape
       #out = out.reshape(B, -1, C)
        for i, blk in enumerate(self.sa_dblock1):
            out = blk(out)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, C, H, W)

        #out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        '''
        #xs = torch.chunk(out, self.shift_size, 1)
        #for i, xin in enumerate(xs):
        for j, conv in enumerate(self.decoder_160):
            fea = conv(out)
            fea = conv(out)
            if j == 0:
                feas = fea
            else:
                feas = torch.cat([fea, feas], dim=1)

            # outs = self.lower_channel1(out)
        feas = channel_shuffle(feas, 3)
        _, _, H1, W1 = feas.shape
        x_win = feas.permute(0, 2, 3, 1).contiguous()
        # x_win = window_shift(x_win, H1 // 4)
        x_win = dilated_shift(x_win, H1)
        #x_win = window_partition(x_win, H1 // 2)
        # _,H2,W2,_=x_win.shape
        # x_win = dilated_shift(x_win, self.dilated_window_size1)
        # x_win = window_partition(x_win, self.window_size1)

        sa_attns = self.sa1(x_win)
        #sa_attns = window_reverse(sa_attns, H1 // 2, H1, W1)

        # sa_attn=window_reverse(sa_attn,self.window_size1,H2,W2)
        sa_attns = dilated_shift_reverse(sa_attns, H1 // 2)
        # sa_attn=window_reverse(sa_attn,self.dilated_window_size1,H1,W1)

        # sa_attn = window_shift(sa_attn, H1 // 4)
        sa_attns = sa_attns.permute(0, 3, 1, 2).contiguous()
        sa_attns = self.ca1(sa_attns)
        sa_attns += feas
        sa_attns = self.lower_channel1(sa_attns)
        out = sa_attns
        out = self.relu3(out)
        B, C, H, W = out.shape
        out = out.reshape(B, -1, C)
        out = self.dnorm2(out)
        out = out.reshape(B, C, H, W)
        out = torch.add(out, t3)
        for i in range(0, 2):
            for j in range(0, 2):

                x_p=out[:,:,32*i:32*(i+1),32*j:32*(j+1)]
                if i==0 and j==0:
                    x_ps=x_p
                else:
                    x_ps=torch.cat([x_ps,x_p],dim=1)
        out=x_ps
        '''
        B, C, H, W = out.shape
        #out = out.reshape(B, -1, C)
        for i, blk in enumerate(self.sa_dblock2):
            out = blk(out)

        out = self.dnorm4(out)
        #out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = out.reshape(B, C, H, W)
        '''
        out = self.relu2(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = self.relu1(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out, t1)
        out = self.relu1(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        #out = F.relu(self.decoderf(out))
        out=self.final(out)

        return out


