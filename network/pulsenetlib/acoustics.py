#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions associated with the problem physics
"""

# native modules
##none##

# third party modules
import torch

# local modules
import pulsenetlib.transforms


def acousticEnergy(p, U, rho0=1, c0sq=340**2):
    r""" Calculates the acoustic energy according to Kirchhoff's equation
    for quiescent fluids (see Rienstra and Hirschberg "An Introduction to
    Acoustics" Chapter 2.7.2).
    
    Careful use this function as it is only valid until strong assumptions 
    (quiescent fluids, reflecting walls, etc).
    
    ----------
    ARGUMENTS
    
        p (Tensor): acoustic pressure field, format (N, 1, D, H, W)
        U (Tensor): input perturbed velocity field, format
                    (N, C, D, H, W) (2 channels in dim 1 if 2D)
        rho0 (float): steady background density
        c0sq (float): squared speed of sound (constant)
    
    ----------
    RETURNS
    
        (Tensor) output scalar acoustic energy density field.
    
    """

    term1 = (0.5/(rho0*c0sq)) * torch.pow(p, 2)
    term2 = 0.5 * rho0 * torch.pow(torch.norm(U, dim=1, keepdim=True), 2)

    return term1.unsqueeze(1) + term2



def energyPreservingCorrection(inputs, estimation, bc=0):
    """ Corrections to preserve the energy

    Set of operations for correcting the estimation in order
    to respect the energy fluxes of the problem.
    Performs in-place operations.

    ----------
    ARGUMENTS

        inputs: tensor with the input density fields,
               format (N, C, D_in, H, W) 
        estimation: tensor with the estimated density field,
                    format (N, C, D_out, H, W)
        bc: selection of the boundary conditions.
            - 0: 4 reflecting walls (default)
            - 1: ...

    ----------
    RETURNS

        ##none## 

    """
    
    if bc == 0:
        """ 4 reflecting walls (no flux of energy),
        Uniform drift based on target's and estimation's average
        density:

             drift = rho_start - rho_estimation
             rho_corrected = rho_estimation + drift

        More details in:

           Alguacil et al. "Predicting the Propagation of Acoustic
           Waves using Deep Convolutional Neural Networks"
           AIAA AVIATION 2020 FORUM. 2020 - doi.org/10.2514/6.2020-2513
        
        """
        avg_rho_inputs = pulsenetlib.transforms.mean_first_frame(inputs)
        avg_rho_estimation = pulsenetlib.transforms.mean(estimation)

        drift = avg_rho_estimation.clone()
        # applying the shift for each estimation frame, based on
        # first input frame
        for index_out in range(estimation.shape[2]):
           drift[:,:,index_out] = \
               avg_rho_inputs[:,:,0,0,0] - avg_rho_estimation[:,:,index_out]
        pulsenetlib.transforms.add_frame_(
           estimation, drift.unsqueeze(-1).unsqueeze(-1), [1])
    else:
        # yet to be implemented!
        pass



