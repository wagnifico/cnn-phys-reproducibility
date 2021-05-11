#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 levels multiscale network

Inputs are shape (batch, channels, in_dim, height, width),
outputs are shape (batch, channels, out_dim, height, width).
Every channel is a physical quantity (density, velocity-x, ...)
and every dimension is a frame in time.

The number of input (data) channels and the number of output
(target) channels are selected at model creation.

"""

# native modules
##none##

# third-party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# local modules
##none##


class _ConvBlock1(nn.Module):
    """ First block - quarter scale.
    
    Four Conv2d layers, all with kernel_size 3 and padding of 1 (padding
    ensures output size is same as input size)
    
    Optional dropout before final Conv2d layer
    
    ReLU after first two Conv2d layers, not after last two - predictions
    can be +ve or -ve
    
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels,
                 out_channels, dropout=False):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=3, padding = 0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid1_channels, kernel_size=3, padding=0),
        ]
        
        if dropout:
            layers.append(nn.Dropout())
        
        layers.append(nn.ReplicationPad2d(1))
        layers.append(
            nn.Conv2d(
                mid1_channels, out_channels, kernel_size=3, padding=0)
                     )
        
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock2(nn.Module):
    """ Second block - half scale.
    
    Six Conv2d layers. First one kernel size 5, padding 2, remainder
    kernel size 3 padding 1.
    
    Optional dropout before final Conv2d layer
    
    ReLU after first four Conv2d layers, not after last two - predictions
    can be +ve or -ve
    
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, mid3_channels,
                 out_channels,dropout=False):
        super(_ConvBlock2, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid3_channels,kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid1_channels, kernel_size=3, padding=0),
        ]
        
        if dropout:
            layers.append(nn.Dropout())
        
        layers.append(nn.ReplicationPad2d(1))
        layers.append(
            nn.Conv2d(
                mid1_channels, out_channels, kernel_size=3, padding=0)
                      )
        
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock3(nn.Module):
    """ Third block - full scale.
    
    Six Conv2d layers. First and last kernel size 5, padding 2, remainder
    kernel size 3 padding 1.
    
    Optional dropout before final Conv2d layer
    
    ReLU after first four Conv2d layers, not after last two - predictions
    can be +ve or -ve
    
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, mid3_channels,
                 out_channels, dropout=False):
        super(_ConvBlock3, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid3_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels, mid2_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels, mid1_channels, kernel_size=3, padding=0),
        ]
        
        if dropout:
            layers.append(nn.Dropout())
        
        layers.append(nn.ReplicationPad2d(2))
        layers.append(
            nn.Conv2d(
                mid1_channels, out_channels, kernel_size=5, padding=0)
                     )
        
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

    
    
class MultiScaleNet(nn.Module):
    """ Define the network.
    
    Input when called is number of input and output channels.
    
        - Downsample input to quarter scale and use ConvBlock1.
        - Upsample output of ConvBlock1 to half scale.
        - Downsample input to half scale, concat to output of ConvBlock1;
          use ConvBLock2.
        - Upsample output of ConvBlock2 to full scale.
        - Concat input to output of ConvBlock2, use ConvBlock3. Output of
          ConvBlock3 has 8 channels
        - Use final Conv2d layer with kernel size of 1 to go from 8
          channels to output channels.
         
    """
    def __init__(self, data_channels, out_channels=1, dropout=False):
        
        super(MultiScaleNet, self).__init__()
        
        self.convN_4 = _ConvBlock1(data_channels,
                                   32, 64, out_channels,
                                   dropout=dropout)
        self.convN_2 = _ConvBlock2(data_channels + out_channels,
                                   32, 64, 128, out_channels,
                                   dropout=dropout)
        self.convN_1 = _ConvBlock3(data_channels + out_channels,
                                   32, 64, 128, 8,
                                   dropout=dropout)
        
        self.final = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, x):
        
        quarter_size = [int(i*0.25) for i in list(x.size()[2:])]
        half_size = [int(i*0.5) for i in list(x.size()[2:])]
        
        convN_4out = self.convN_4(
            F.interpolate(x,(quarter_size),
                          mode="bilinear",
                          align_corners=True)
                                 )
        
        convN_2out = self.convN_2(
            torch.cat((F.interpolate(x,(half_size),
                                     mode="bilinear",
                                     align_corners=True),
                       F.interpolate(convN_4out,(half_size),
                                     mode="bilinear",
                                     align_corners=True)),
                      dim=1)
                                 )
        
        convN_1out = self.convN_1(
            torch.cat((F.interpolate(x,(x.size()[2:]),
                                     mode="bilinear",
                                     align_corners=True),
                       F.interpolate(convN_2out,(x.size()[2:]),
                                     mode="bilinear",
                                     align_corners=True)),
                      dim=1)
                                 )
        
        final_out = self.final(convN_1out)
        
        return final_out
