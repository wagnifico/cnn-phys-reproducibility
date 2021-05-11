#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical functions
"""

# native modules
##none##

# third-party modules
import torch

# local modules
##none##


def first_derivative(field, infinitesimal=1, transpose=False):
    """ Function for calculating the 4-order central difference
    derivative (uniform grid).

    The function always calculates the gradient based on the last
    dimension (columns, x derivative). For derivating in row
    direction (y derivative), transpose the array (set transpose
    to True).
    
    TODO: the transposing operations are not necessary!
    maybe it can be costy! to investigate

    ----------
    ARGUMENTS
    
        field: 2D scalar field in form of a tensor of shape
               (N, C, D, H, W)
        infinitesimal: mesh infinitesimal. Optional, 1 as default.
        transpose: transpose the input and output for perfoming
                   the gradient for the rows (y). Optional, False
                   as default.
    ----------
    RETURNS
    
        derivative: tensor with the 1st derivative, in the same shape
                    as the input (N, C, D, H, W)
        
    """
    
    if transpose:
        field = field.transpose(3,4)
    
    # allocating the derivative at the same device as the input field
    derivative = torch.zeros(field.shape, device=field.device)
    
    # calculating the centered finite difference, 4th order
    # elements affected by the border are overwritten later
    derivative[:,:,:,:,1:-1] = (2/3)*(-field[:,:,:,:,:-2] + field[:,:,:,:,2:])
    derivative[:,:,:,:,2:-2] = derivative[:,:,:,:,2:-2] + \
        (1/12)*(field[:,:,:,:,:-4] - field[:,:,:,:,4:])
    
    # border (forward and backward, degenerated to 2nd order)
    derivative[:,:,:,:,0] = \
        (-3/2)*field[:,:,:,:,0] + (2)*field[:,:,:,:,1] + (-1/2)*field[:,:,:,:,2]
    derivative[:,:,:,:,1] = \
        (-3/2)*field[:,:,:,:,1] + (2)*field[:,:,:,:,2] + (-1/2)*field[:,:,:,:,3]
    derivative[:,:,:,:,-1] = \
        (1/2)*field[:,:,:,:,-3] + (-2)*field[:,:,:,:,-2] + (3/2)*field[:,:,:,:,-1]
    derivative[:,:,:,:,-2] = \
        (1/2)*field[:,:,:,:,-4] + (-2)*field[:,:,:,:,-3] + (3/2)*field[:,:,:,:,-2]

    if transpose:
        derivative = derivative.transpose(3,4).contiguous()
    
    return derivative/infinitesimal



def derivate_dx(x, dx):
    """ Wrapper function to calculate the derivative in x
    """
    return first_derivative(x, infinitesimal=dx)



def derivate_dy(x, dx):
    """ Wrapper function to calculate the derivative in y
    """
    return first_derivative(x, infinitesimal=dx, transpose=True)



def divergence(x, dx):
    """ Calculate the divergence field of x, considering uniform
    infinitesimal dx
    """
    return derivate_dx(x, dx) + derivate_dy(x, dx)



def online_mean_and_sd(loader):
    """ Compute the first and second statistical moments

                Var[x] = E[X^2] - E^2[X]
                
    ----------    
    ARGUMENTS

        loader: dataloader of the model
        
    ----------    
    RETURNS

        fst_moment: first moment in format (C,3), each row
                    representing a channel (quantity)
        second moment

    """

    # reading a case just to define the dimensions
    dummy_data, _, = next(iter(loader))
    _, c, t, h, w = dummy_data.shape
        
    cnt = torch.zeros((c,3))
    fst_moment = torch.empty((c,3))
    snd_moment = torch.empty((c,3))
    
    for h_data, __ in loader:
        b, _, _, _, _ = h_data.shape
        # number of pixels per channel (quantity)
        nb_pixels = b * t * h * w

        if torch.cuda.is_available():
            data = h_data.cuda(non_blocking=True)
        else:
            data = h_data
        
        # calculate the derivatives of each page
        gdl_x = derivate_dx(data, dx=1.)
        gdl_y = derivate_dy(data, dx=1.)
        
        for i, field in enumerate([data, gdl_x, gdl_y]):
            for j in range(c):
                sum_ = torch.sum(field[:,j])
                sum_of_square = torch.sum(field[:,j] ** 2)

                fst_moment[j,i] = \
                    (cnt[j,i]*fst_moment[j,i] + sum_) / \
                        (cnt[j,i] + nb_pixels)
                
                snd_moment[j,i] = \
                    (cnt[j,i]*snd_moment[j,i] + sum_of_square) / \
                        (cnt[j,i] + nb_pixels)

                cnt[j,i] += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)



def GDL(prediction, target, dx, alpha=2, reduction='mean'):
    """ Function to combine gradients into gradient difference loss
    ----------
    ARGUMENTS
    
        prediction: tensor with the model's output, format 
        target: tensor with target 
        dx: grid element size
        alpha: exponential of the error. Optional, default is 2 (L2 norm)
        reduction: string selecting between the sum of mean of the error.
                   Optional, default is mean. If 'none', error arrays
                   are returned.
        
    ----------
    RETURNS
    
        loss: error array/value
    
    """
    
    loss = torch.abs(
        derivate_dx(prediction,dx) - derivate_dx(target,dx))**alpha
    loss += torch.abs(
        derivate_dy(prediction,dx) - derivate_dy(target,dx))**alpha
    
    if reduction != 'none':
        loss = torch.mean(loss) if reduction == 'mean' \
            else torch.sum(loss)
                
    return loss



if __name__ == '__main__':
    
    print("Testing the derivation code")
    # Testing the derivation
    
    # generating the uniform grid
    grid_size = 500
    x = torch.linspace(0, 100, steps=grid_size)
    y = torch.linspace(0, 100, steps=grid_size)
    
    dx = x[1] - x[0]
    
    # using reverse (considering rows as y and columns as x)
    grid_y, grid_x = torch.meshgrid(x, y)
    
    # defining the test equation
    z = 4*grid_y + 5*grid_x**2 + 5*grid_x*torch.sin(grid_y)
    
    # analytical derivatives
    analytical_dzdx = 10*grid_x + 5*torch.sin(grid_y)
    analytical_dzdy = 4 + 5*grid_x*torch.cos(grid_y)
    
    # adding dimensions to use on the functions
    z_torch = z.unsqueeze(0).unsqueeze(0)
        
    criterion = torch.nn.MSELoss()
    
    # calculating the error    
    error_dzdx = criterion(analytical_dzdx,
                           derivate_dx(z_torch,dx)[0,0])
    error_dzdy = criterion(analytical_dzdy,
                           derivate_dy(z_torch,dx)[0,0])
        
    print("\nDerivative error: {:.5e} {:.5e}\n".format(
                error_dzdx,error_dzdy))
    
