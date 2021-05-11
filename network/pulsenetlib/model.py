#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of the neural network model
"""

# native modules
##none##

# third-party modules
import torch.nn as nn

# local modules
import pulsenetlib


class PulseNet(nn.Module):
    """ Class of the neural network model
    
    Currently, only considers the 2D convolution network.
    
    
    ----------    
    ARGUMENTS
    
        number_channels: number of channels that are considered
                         by the model (data fields)
        input_frames: number of frames of the input (number of 
                      frames)
        output_frames: number of frames of the output/target                 
        std_norm: tensor with the normalization factors (must have
                  as much values as channels - second dimension
                  of the input)        
        avg_remove: tensor with the average removal factors (must
                    have as much values as channels - second dimension
                    of the input)        
        dropout: boolean for setting dropout technique.
                 Optional, default is True
        
    """
    def __init__(self, number_channels, input_frames,
                 output_frames, std_norm, avg_remove,
                 dropout=True):
        super(PulseNet, self).__init__()
        
        # training parameters
        self.num_output_frames = output_frames
        self.num_pages = input_frames*number_channels
        self.std_norm = std_norm
        self.avg_remove = avg_remove
        
        # defining the neural network: MultiScaleNet
        self.net = pulsenetlib.neuralnet2d.MultiScaleNet(
                            self.num_pages, number_channels)
        
        
        
    def forward(self, input_):
        """ Method to call the network and evaluate the model.
        
        Both the inut and the output are in physical units.
        It contains the data treatement (normalization,
        multiplication, etc) for before and after the network.
        
        ----------
        ARGUMENTS
        
            input_: tensor in the format (N, C, D_in, H, W)
                N: number of batch
                C: channel (physical quantity)
                D_in: frame (timestep)
                H: frame height
                W: frame width
        
        ----------
        RETURNS
        
            out: tensor with the prediction, of the same format
                 single frame (N, C, D_out, H, W)
        
        """
        
        # input format
        bsz, n_channels, n_frames, h, w = input_.size()
        
        
        # calculate the std of the first frame (timestep) of every
        # element in the input scene
        std_first_frames = \
            pulsenetlib.transforms.std_first_frame(input_)  
        mean_first_frames = \
            pulsenetlib.transforms.mean_first_frame(input_)  
        
        
        # remove the average, considering the first frame
        pulsenetlib.transforms.add_frame_(
            input_, mean_first_frames, self.avg_remove)
        # normalize by std, considering the first frame
        pulsenetlib.transforms.divide_frame_(
            input_, std_first_frames, self.std_norm)
        
        
        # for performing a 2D convolution, the data must be 3D.
        # This is achieved by combining dimensions 2 and 3
        # (channel and frame, respectively). Data is presented 
        # to the neural network as:
        #
        #     channel 0: frame 0, channel 1: frame 0, ....
        #       channel 0: frame 1, channel 1, frame 1, ....
        #         channel 0: frame 2, channel 1, frame 2, ....
        #           ...
        #
        # the transpose operation is necessary so they are sorted
        # by time and not by channel
        x = input_.transpose(1,2).contiguous().view(
                                        bsz,self.num_pages,h,w)
        # calling the network
        y = self.net(x)
        
        
        # returning to original data format!
        # first as (N, D, C, H, W), later modified to (N, C, D, H, W)
        # by the transpose method
        out = y.view(
            bsz, self.num_output_frames, n_channels, h, w).transpose(1,2)
        
        
        # going back to physical quantity performing the opposite
        # operations
        pulsenetlib.transforms.multiply_frame_(
            out, std_first_frames, self.std_norm)
        pulsenetlib.transforms.add_frame_(
            out, mean_first_frames, -self.avg_remove)
        
        
        return out
    
    
