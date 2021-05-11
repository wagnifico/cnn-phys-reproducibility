#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for plotting the estimated fields and associated errors

"""

# native modules
import glob
import copy
import warnings

# third-party modules
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches

# local modules
    

class RecursivePanelPlot():
    """ Panel with the time evolution of 2D fields.

    Default inputs will produce a plot with the first row as the
    reference (simulated data), second row for estimations and third
    row for error.
    
    A path is added at the start of the plot such as to hide
    frames not issued from a run of the model.
    
    TODO: Add an axes for the colorbar!

    ----------
    ARGUMENTS

        nx, ny: field height and width
        num_input_frames: number of input frames
        num_output_frames: number of output frames
        num_rows: number of quantities (rows) to be plotted. Optional, 
                  default is 3 (reference, estimation and square error)  
        title_fun: function that returns an string used as row title based
                   on the plot index (from 0 to the number of total frames).
                   Normally associated with the number of the frames or
                   timestep. Optional, default returns an empty title
    ----------
    RETURNS
    
        ##none##

    """
    
    def default_title(self,args):
        return ''

    def __init__(self, nx, ny, num_input_frames, num_output_frames,
                 title_fun=None,
                 num_rows=3,
                 colormaps=('seismic','seismic','hot'),
                 limits=([-8e-4, 8e-4],[-8e-4, 8e-4],[1e-10,1]),
                 labels=('$\\mathrm{target}$\n$f(t)$',
                         '$\\mathrm{estimation}$\n$\\tilde{f}(t)$',
                         r'$\dfrac{[f(t) - \tilde{f}(t)]^2}{\mathrm{max}(f(t))}$'),
                 log_colorbar=(False,False,True),
                 cover_row=(False,True,True),
                ):
        
        if title_fun is None:
            self.title_fun = self.default_title
        else:
            self.title_fun = title_fun 
        
       
        self.input_frames = num_input_frames
        self.cover_row = cover_row
        self.log_colorbar = log_colorbar
        
        total_number_frames = num_input_frames + num_output_frames
        
        self.fig, self.axs = plt.subplots(
            num_rows, total_number_frames + 1,
            gridspec_kw={'width_ratios': [*([1]*total_number_frames), .25],
                         'height_ratios': [1]*num_rows},
            figsize=(total_number_frames, num_rows),
            dpi=150)

        dummy_data = np.zeros((nx, ny))
        self.plot_objs = np.zeros(self.axs.shape,dtype=object)
        self.plot_patches = self.plot_objs.copy()

        # colorbar axis
        self.caxs = np.zeros((num_rows),dtype=object)
        self.cbars = self.caxs.copy()

        # setting format of axis labels and starting plots
        for index_row, cmap, limit, label in zip(
                                range(num_rows), colormaps, limits, labels):

            for index_col in range(total_number_frames):
                ax = self.axs[index_row, index_col]
                # remove only ticks labels
                ax.set_xticklabels([]), ax.set_yticklabels([])
                # remove ticks
                ax.set_xticks([]), ax.set_yticks([])            
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(0.25)

                # starting plots
                self.plot_objs[index_row, index_col] = \
                            ax.imshow(dummy_data,
                                      origin='lower', interpolation='none',
                                      cmap=cmap, vmin=limit[0], vmax=limit[1])

            # colorbar on last axes of row
            self.caxs[index_row] = self.axs[index_row, -1]
            self.cbars[index_row] = self.fig.colorbar(
                                        self.plot_objs[index_row, -2],
                                        cax=self.caxs[index_row],
                                        orientation='vertical',
                                        shrink=.5,
                                        #extend='both'
            )
            self.caxs[index_row].set_aspect(10)
            self.cbars[index_row].ax.tick_params(labelsize=5) 
            for axis in ['top','bottom','left','right']:
                self.caxs[index_row].spines[axis].set_linewidth(0.25)

            # add a label at the left of the first field
            self.axs[index_row, 0].annotate(label,
                                       xy=(int(-.2*nx), ny/2),
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       annotation_clip=False,
                                       rotation=90,
                                       fontsize=7)

        # covering non-pertinent fields. make sure the box color is
        # not present in the colormap to avoid confusion and to close
        # the patches later. Patches are created for all fields, but
        # removed for quantities that are not hidden
        for index_row, cover in enumerate(cover_row):
            for index_col in range(num_input_frames):            
                self.plot_patches[index_row, index_col] = \
                    self.axs[index_row, index_col].add_patch(
                        patches.Rectangle((0, 0), nx, ny, facecolor='gray')
                    )
                if not cover:
                    self.plot_patches[index_row, index_col].remove()

        # logarithmic colormap for the error plot
        for index_row, is_log in enumerate(log_colorbar):
            if is_log:
                for index_col in range(total_number_frames):
                    self.plot_objs[index_row, index_col].set_norm(
                        matplotlib.colors.LogNorm(
                            vmin=limits[index_row][0],
                            vmax=limits[index_row][1],
                            clip=True)
                    )

        def remove_last(array):
            """ Function to remove last element of numpy array.
            One must create a new array because numpy arrays are immutable.
            """
            array_temp = np.zeros(
                (array.shape[0], array.shape[1]-1), dtype=object)
            for index_row in range(array.shape[0]):
                array_temp[index_row,:] = array[index_row,:-1]
            return array_temp
        
        self.axs = remove_last(self.axs)
        self.plot_objs = remove_last(self.plot_objs)

        plt.tight_layout(pad=1, h_pad=0, w_pad=0)
        self.fig.subplots_adjust(wspace=0.1, hspace=0.1)


        
    def update(self, cases, counter):
        """ Update the plotted fields
        
        ----------
        ARGUMENTS
        
            cases: tuple with the (N, H, W) tensors with the values
                   to be plotted on each row
            counter: integer indicating the number of recurrencies.
                     Used to remove the patches hiding non-pertinent
                     fields
        
        ----------
        RETURNS
        
            ##none##
        
        """
        
        for index_row, case in enumerate(cases):
            for index_plot, field in enumerate(case):
                plot_obj = self.plot_objs[index_row, index_plot]
                data_to_plot = field.data.numpy()
                
                # to avoid misconceptions and errors with matplotlib,
                # forcing null values to the limit of colorbar
                # in case of a log scale
                if self.log_colorbar[index_row]:
                    data_to_plot[data_to_plot <= 0] = 1e-30      
                plot_obj.set_data(data_to_plot)
                
                # add title on first row
                if index_row == 0:
                    self.axs[0, index_plot].set_title(
                        self.title_fun(index_plot),
                        fontdict={'fontsize':8})

        # removing patches covering non-pertinent fields, that is,
        # before any estimation
        if counter < self.input_frames + 1 and counter > 0:
            index_to_remove = self.input_frames - counter
            for index_row, cover in enumerate(self.cover_row):
                if cover:
                    self.plot_patches[index_row, index_to_remove].remove()

            

    def save(self, filepath):

        def reduce_range():
            # Due to an unkwon behavior, in the case of values increasing
            # and starting to be out of the LogNorm range (but still positive),
            # the imshow plot returns an error:
            #
            # >>> ValueError: minvalue must be positive
            #
            # Current workaround consists in change the range of the plot.
            # This is not a reasonable solution once the generated images
            # will not be directly comparable to the previous plots and may
            # cause confusion.
            # 
            # The operation is performed for all the rows with log scales, even
            # if they do not present the error.
            #
            # TODO: find a better solution and/or report the bug to matplotlib
            # devellopers
            range_factor = 1e3
            for images, is_log in zip(self.plot_objs, self.log_colorbar):
                for image in images:
                   if is_log:
                       vmin, vmax = image.get_clim()
                       warnings.warn(
                           'Updating the range of the image due to LogNorm  bug')
                       image.set_norm(
                          matplotlib.colors.LogNorm(
                              vmin=vmin*range_factor, vmax=vmax*range_factor)
                       )
                       #image.set_clim(vmin*range_factor, vmax*range_factor)
            plt.draw()

        try:
            plt.draw()
        except ValueError:
            reduce_range()
           
        self.fig.savefig(filepath)
