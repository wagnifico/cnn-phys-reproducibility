#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Classes that define the loss criterion used for training the
model.

"""

# native modules
import functools

# third-party modules
import torch
import torch.nn as nn

# local modules
import pulsenetlib


class Criterion(nn.Module):    
    def __init__(self, L2lambda, GDL2Lambda, run_device):
        """ Class that defines the loss functions.

        For instanciating, one most indicates the gain (lambda) for
        each component (L2 and GDL2), representing, respectively,
        the L2-norm and L2-norm of the gradient. If any of the lambdas is
        null, the corresponding loss is not calculated.

        TODO: make it channel independent (for example, 0.5 for P,
        0.25 for Ux and 0.25 for Uy

        ----------
        ARGUMENTS

            L2lambda: lambda of the L2-norm
            GDL2Lambda: lambda of gradient L2-norm
            run_device: running device
        
        ----------
        RETURNS
        
            ##none##
            
        """
                
        super(Criterion, self).__init__()

        # loss types
        self._mse = nn.MSELoss()
        
        # define the functions that calculate the loss forms
        # only selected if the multiplier is bigger than 0
        self.loss_funs = []
        self.loss_names = []
                
        if L2lambda > 0.0:
            self.loss_names.append('L2Lambda')
            self.loss_funs.append(
                functools.partial(self.loss_L2, lamb=L2lambda)
                                 )
                
        if GDL2Lambda > 0.0: 
            self.loss_names.append('GDL2Lambda')   
            self.loss_funs.append(
                functools.partial(self.loss_GDL2, lamb=GDL2Lambda)
                                 )
        
        # error when no loss term has been defined
        if len(self.loss_funs) == 0:
            raise ValueError('At least one non-null loss term is necessary.')

        # Normalization by the number of loss components
        self.normalization = torch.ones(
            len(self), device=run_device)  

    def get_loss_names(self):
        return self.loss_names 
    
    def __len__(self):
        """ Returns the number of loss terms.
        """
        return len(self.loss_funs)
    
    def summary(self):
        """ Prints the loss terms functions.
        """
        print('Selected loss terms:\n')
        for loss_fun in self.loss_funs:
            print(loss_fun,'\n')
    
    
    
    def forward(self, out, target):
        """ Calculate the loss.
        ----------
        ARGUMENTS
        
            out: model prediction tensor, format (N,C,D,H,W)
            target: target tensor, format (N,C,D,H,W)
            normalization: normalization factor
        ----------    
        RETURNS
        
            result: loss tensor, normalized by the
                    normalization factor
        
        """
                
        _assert_no_grad(target)

        if self.normalization.size(0) != len(self):
            raise ValueError(
                'Mismatch between size of normalization ({})\
                and number of losses ({}).'.format(
                    self.normalization.size(0),
                    len(self.loss))
            )

        # calculate the loss term for every available function
        result = [loss_fun(out, target)/norm for loss_fun, norm in 
                zip(self.loss_funs, self.normalization)]
        return result
    
    
    
    def loss_L2(self, prediction, target, lamb=0.0):
        """ Calculate the loss of the fields
        
        Each frame is normalized by the standard deviation
        observed for each channel in the target's first time 
        frame (:,:,0,:,:).
        
        ----------
        ARGUMENTS
        
            prediction: tensor with the model's output, format
                        (N, C, D, H, W)
            target: tensor with the target, format 
                    (N, C, D, H, W)
            lamb: scalar that indicates the gain of the term in
                  the total loss calculation
        
        ----------
        RETURNS
            
            (loss)
            
        """
        
        std_mse = pulsenetlib.transforms.std_first_frame(target)
        
        return lamb*self._mse(
            prediction/std_mse, target/std_mse)
    
    
    
    def loss_GDL2(self, prediction, target,lamb=0.0):
        """ Calculate the loss of the gradients
        
        Deals with a single channel.
        
        ----------
        ARGUMENTS
        
            prediction: tensor with the model's prediction,
                        format (N, C, D, H, W)
            target: tensor with the target, format
                    (N, C, D, H, W)
            lamb: scalar that indicates the gain of the term in
                  the total loss calculation
            
        ----------
        RETURNS
            
            (loss)
            
        """
        
        # calculating the target gradients in x and y
        grad_x_tar = pulsenetlib.tools.derivate_dx(target, dx=1.)
        grad_y_tar = pulsenetlib.tools.derivate_dy(target, dx=1.)
        
        std_grad_x_first_frame = \
            pulsenetlib.transforms.std_first_frame(grad_x_tar)
        std_grad_y_first_frame = \
            pulsenetlib.transforms.std_first_frame(grad_y_tar)
        
        # selecting the max grad std for each channel for 
        # normalization
        std_grad = \
            torch.max(std_grad_x_first_frame, std_grad_y_first_frame)
        
        return lamb*pulsenetlib.tools.GDL(
            prediction/std_grad, target/std_grad, alpha=2., dx=1.)

    
    
def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"
