#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scripts for testing a trained model based on comparing it
to reference solutions

Due to python's limitation in picking functions and modules
that are not in high-level, the classes here consider the model
as in the current pulsenetlib, and not as saved on the model directory.


TODO: make it work with the model as loaded from the local
      model module copy
TODO: adapt to run on GPU (using a batch of cases!)
TODO: implement a parallelism that is compatible with GPU


"""

# native modules
import os
import sys
import datetime
import warnings

# third-party modules
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI

# local modules
import pulsenetlib.datatransfer


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == 0


class RecursiveTester():
    """ Testing the model recursively
    
    This class allows to test the models response recursively, that is,
    the output of the model from previous recursions is used as an input
    
    ----    
    ARGUMENTS

        model_path: folder where the configuration and state files are stored
        run_device: device where to run the testing
        comparator: comparator instance which contains the operations used to
                    generate the comparison fields and stats from target and 
                    output tensors
        reference_path: folder where the test cases are stored (root folder
                        of multiple scenes sub-folders)
        checkpoint_file_path: path to checkpoint file to consider. 
                              Optional, if not given (None), latest checkpoint
                              in default folder will be used
        file_format: format of the raw simulation file
        apply_EPC: boolean indicating the the energy preserving correction
                   should be applied. More info in pulsenetlib.acoustics.
                   Optional, default is True
        comparison_name: string with the name of the comparison
                         (to be added on files). Optional, default
                         is 'test'.
                 
    """
    
    def __init__(self, model_path, run_device, comparator,
                 reference_path, file_format,
                 checkpoint_file_path=None,
                 apply_EPC=True, 
                 comparison_name='test'):
        
        if checkpoint_file_path is None:
            checkpoint_file_path='./checkpoints/epoch=*.ckpt'

        # read configuration files and loading model from the folder        
        self.net, state_dict, train_configuration, model_configuration = \
            pulsenetlib.datatransfer.load_model(
                model_path,
                ckpt_format=checkpoint_file_path,
                run_device=run_device)
        
        self.run_device = run_device

        self.avg_remove = model_configuration['avgRemove']
        self.std_norm = model_configuration['stdNorm']
        
        self.channels = model_configuration['channels']
        self.number_channels = len(self.channels)
        self.input_frames = model_configuration['numInputFrames']
        self.output_frames = model_configuration['numOutputFrames']
        self.total_number_frames = self.input_frames + self.output_frames
        self.scalar_add = torch.tensor(
            model_configuration['scalarAdd']).to(device=run_device)
        self.frame_step = train_configuration['frameStep']
        
        self.net.eval()
                
        self.ref_database_path = reference_path
        self.ref_file_format = file_format        
        path_ref_files = os.path.join(reference_path, file_format)

        self.comparator = comparator

        # results are saved on subfolder of models'
        self.results_path = os.path.join(
            model_path, 'results/' + comparison_name)
        if not os.path.isdir(self.results_path):
            os.makedirs(self.results_path, exist_ok=True)

        self.figures_path = os.path.join(self.results_path, 'figs')
        if not os.path.isdir(self.figures_path):
            os.makedirs(self.figures_path, exist_ok=True)

        self.VTK_path = os.path.join(self.results_path, 'vtk')
        if not os.path.isdir(self.VTK_path):
            os.makedirs(self.VTK_path, exist_ok=True)

        self.apply_EPC = apply_EPC
        self.comparison_name = comparison_name
        
        # reading the files
        _, list_files, array_references = \
            pulsenetlib.dataset.list_available_files(path_ref_files)

        # selecting the files to consider based on the selected and
        # available frame steps - supposes all scenes (simulations) have
        # the same time step!
        available_time_step = int(
            array_references[1][1] - array_references[0][1])
        if self.frame_step % available_time_step == 0:
            # selecting the multiples of the available files
            temp_list_raw_files = []
            jump_step = int(self.frame_step // available_time_step)
            for scenes_files in list_files:
                temp_list_raw_files.append(scenes_files[::jump_step])

            list_files = temp_list_raw_files
            del temp_list_raw_files

        self.list_files = list_files


        
    def run(self, channel_index=0, 
            frame_start=0, frames_to_estimate=0,
            number_workers=12, plot_panel=True, plot_graphs=True,
            export_field=False,
            verbose=True):
        """ Perform the testing
        
        -----------
        ARGUMENTS
        
            channel_index: index of channel (quantity) to be analysed,
                           according to the second dimension, Optional, 
                           default is 0 (density)
            frame_start: index of the first frame to consider. Optional, 
                         default is 0 (start at the first frame possible) 
            frames_to_estimate: number of frames to be analysed (number of 
                                recurrencies). Optional, default is 0 (the 
                                analysis is performed for all files available 
                                in test folder)
            number_workers: number of jobs for performing the test. Optional,
                            default is 12
            plot_panel: boolean to select if plotting the comparison panel,
                        defined in the Comparitor. Optional, default is True
            plot_graphs: boolean to select if a graph with the evolution
                         of the losses/erros is saved. Optional, default is
                         True
            export_field: boolean for activating export estimated field in VTK
                          format. Optional, default is False.
            verbose: boolean to print calculation logs. Optional, True
                     as default
            
        ----------
        RETURNS

            ##none##

        """
        
        # setting up a criterion class that will account for both the
        # field and its gradient. Use values inside model configuration
        # dict to reproduce the training
        self.criterion = pulsenetlib.criterion.Criterion(
            0.98, 0.02, self.run_device)
        self.criterion.normalization = torch.tensor([1.0,1.0])
        self.losses_names = ('L2-norm','grad L2-norm')

        self.channel_index = channel_index
        if frame_start == 0: frame_start = self.frame_step*self.input_frames
        self.frame_start = frame_start
        self.frames_to_estimate = frames_to_estimate
        self.plot_panel = plot_panel
        self.plot_graphs = plot_graphs
        self.export_field = export_field
        self.verbose = verbose
        
        comm.Barrier()
        if is_root:
            if number_workers != size:
                warnings.warn('Number of MPI processes is different than ' +
                            'the number of selected workers.\nSetting number '
                            'of workers to {:d}.'.format(size))
                number_workers = size

            print('\nTesting model database:\n  {:}'.format(
                self.ref_database_path))    
            print('\nNumber of selected scenes: {:d}'.format(
                len(self.list_files)))
            print('Number of workers: {:d}\n'.format(number_workers))
            try:   
                import psutil
                print('memory use: {:.1f}%'.format(
                    psutil.virtual_memory().percent))
            except ImportError:
                pass
        sys.stdout.flush()

        inputs = []
        for index_scene, scenes_files in enumerate(self.list_files):
            inputs.append([index_scene, scenes_files])

        if is_root:
            start_time = datetime.datetime.now()
            print(start_time,'\n')
            if number_workers > 1:
                print('Preparing parallel run...')
                # splitting the inputs (scenes) to be tested between 
                # the processes. Excedents are later spllited between
                # the last groups of scenes

                # TODO: send empty inputs to excedent procs to avoid giving
                #       an error
                if size > len(inputs):
                    raise ValueError(
                        'Number of workers is bigger than number of cases. ' +
                        'Re-call script with an smaller number of processes')

                chunk_size = len(inputs) // size 
                inputs = [inputs[i:i+chunk_size] 
                    for i in range(0, len(inputs), chunk_size)]
                destination_index = -2
                while len(inputs) > size:
                    for index, inp in enumerate(inputs[-1]):
                        inputs[destination_index-index] = \
                            [*inputs[destination_index-index],inp]
                    inputs.pop()
                    destination_index -= 1
                print('Number of scenes per proc: ',
                        *[len(inp) for inp in inputs])
                print('')
            else:
                inputs = [inputs]
        
         
        cases = comm.scatter(inputs, root=0)
        counter = 0
        for case in cases:
            self.step(case)
            # restarting the comparator plot instance
            self.comparator.reset()
            counter += 1
            counters = comm.gather(counter, root=0)
            if not self.verbose and is_root:  
                print('  {:06d}/{:06d}'.format(
                            np.sum(np.array(counters)),
                            len(self.list_files)
                            ),
                      flush=True,
                    )

        comm.Barrier() 
        if is_root:
            #comm.Barrier()
            print('\nDone\n')
            end_time = datetime.datetime.now()
            print(end_time,'\n')
            print('Ellapsed time:',end_time - start_time)
        
        
    def step(self, case):  
        """ Method defining the comparison operation performed for
        each scene (simulaton)

        ----------
        ARGUMENTS
        
            case: tuple of values:
                - index_scene: integer indicating the scene (simulation)
                               to analyze
                - scene_files: list with the paths of the simulation
                               files to be read
        
        ----------
        RETURNS
 
            ##none##        

        """
                
        index_scene, scenes_files = case

        available_frames = len(scenes_files)
        scene_name = '{:}_{:06d}'.format(self.comparison_name, index_scene)
        if self.apply_EPC: scene_name += '_EPC'
 
        vtkloader = pulsenetlib.preprocess.VTKLoader(self.channels)
        _, nx, ny = vtkloader.load2D(scenes_files[0])

        # initial group os frames, shape: (N, C, D, H, W)
        data = torch.zeros((1, self.number_channels, self.input_frames,
                            nx, ny),
                           requires_grad=False)

        index_file = self.frame_start // self.frame_step
        for index_data, i_file in enumerate(
            index_file + np.arange(-self.input_frames,0)):
                data[:,:,index_data,:,:], _, _ = \
                    vtkloader.load2D(scenes_files[i_file])

        # data preparation is performed (for example, using
        # acoustic density instead of density)
        pulsenetlib.transforms.offset_(data, self.scalar_add)

        # reference frames
        # even if not necessary, the reference tensor reproduces the
        # original format for consistency of the coding
        reference = data.clone()

        # adding the last frame
        for i in range(self.output_frames):
            reference = torch.cat(
                (reference,
                 reference[:,:,self.channel_index,:,:].unsqueeze(2)
                 ),
                2)

        initial_comparisons, _ = self.comparator(
            data, reference.clone()[:,:,:-self.output_frames])
        
        comparisons = [reference.clone()]*len(initial_comparisons)        
        for index, comparison in enumerate(initial_comparisons):
            comparisons[index][:,:,:-self.output_frames] = comparison
           
        estimation = torch.zeros(
            (1, self.number_channels, self.output_frames, nx, ny), 
            requires_grad=False
        )

        flatten = lambda l: [item for sublist in l for item in sublist]
        
        header_format, data_format = '{:^4} {:^6} {:^8}', '{:4d} {:6d} {:8d}'
        for error in flatten(self.comparator.names):
            header_format += ' {:^10}'; data_format += ' {:^10.3e}'

        if self.plot_panel:
            title_fun = lambda index_plot: '{:d}'.format(
                frame_number - self.frame_step*(
                    self.total_number_frames - index_plot - 1)
            )
            self.comparator.panel.title_fun = title_fun
             
        if self.verbose and is_root:
            print(header_format.format(
                '#', 'group', 'frame', *flatten(self.comparator.names)))

        errors = [[] for group in self.comparator.names]
        losses, frames = [], []
        frame_number = self.frame_start

        if self.frames_to_estimate == 0:
            end_file = available_frames
        else:
            end_file = index_file + self.frames_to_estimate

        frame_counter = 0
        while index_file < end_file:

            target, _, _ = vtkloader.load2D(scenes_files[index_file])
            target = target[(None,)*2]
            pulsenetlib.transforms.offset_(target, self.scalar_add)
            
            # run the model
            with torch.no_grad():
                output = self.net(data.clone())
            
            detailed_loss = self.criterion(output, target)
            losses.append(torch.tensor(detailed_loss))

            # update the fields
            reference[:,:,-self.output_frames:,:,:] = target
            estimation = torch.cat((data, output), 2)
                                   
            # apply energy correction
            if self.apply_EPC:
                pulsenetlib.acoustics.energyPreservingCorrection(
                   data, estimation, bc=0)
                output = estimation[:,:,-self.output_frames:]
            
            # calculate the comparison
            new_comparisons, comparison_errors = \
                self.comparator(target, output)
            for index, comparison in enumerate(new_comparisons):
                comparisons[index][:,:,-self.output_frames:] = comparison

            for err, esti_err in zip(errors, comparison_errors):
                err.append(esti_err)

            if self.plot_panel:
                self.comparator.panel.update(
                    (reference[0, self.channel_index],
                     estimation[0, self.channel_index],
                     *[comp[0, self.channel_index] for 
                           comp in comparisons]),
                     frame_counter
                )
                self.comparator.panel.save(os.path.join(
                        self.figures_path,'{:}_frame{:05d}.png'.format(
                            scene_name, frame_number))
                )               

            if self.export_field:
                VTK_filename = os.path.join(
                        self.VTK_path,'vtk_{:}_frame{:05d}.vti'.format(
                            scene_name, frame_number)
                            )
                save_VTK_2D(VTK_filename,
                            estimation[0, self.channel_index, -1, :, :],
                            'density',
                            dx=1.0,dy=1.0)

            # rolling data, reference and error tensors
            data = torch.roll(data, -self.output_frames, dims=2)
            data[:,:,-self.output_frames:,:,:] = output
            
            reference = torch.roll(
                reference, -self.output_frames, dims=2)
            for index, comparison in enumerate(comparisons):
                comparisons[index] = torch.roll(
                    comparison, -self.output_frames, dims=2)
           
            if self.verbose:
                print(
                    data_format.format(
                        index_scene, frame_counter, frame_number,
                        *flatten(comparison_errors)),
                    flush=True
                )

            index_file += 1
            frame_counter += 1
            frame_number += self.frame_step
            frames.append(frame_number)

        if self.plot_panel:
            plt.close()

        # graphs with the evolution of the errors and losses
        for name, array, labels in zip(
            ('loss', *self.comparator.group_names),
            (losses, *errors),
            (self.losses_names, *self.comparator.names)):

            case_name = self.comparator.filename_format.format(
                 scene_name, name)
            graph_name = 'graph_{:}.png'.format(case_name)

            # convert list of tensors into numpy array
            array_tensor = torch.tensor([], requires_grad=False)
            torch.cat(array, out=array_tensor)
            array = array_tensor.reshape((len(array), -1)).numpy()

            if self.plot_graphs:
                fig, ax = plt.subplots(figsize=(5,5))
                for index in range(array.shape[1]):
                    ax.semilogy(np.array(frames), array[:,index],
                                label=labels[index],
                                linewidth=.75)
                ax.set_xlabel('frame')
                ax.set_ylabel(name)
                ax.legend()
                ax.grid(True, which='both',
                        linewidth=.5, alpha=.5, color='gray')
                plt.tight_layout()    
                plt.savefig(
                    os.path.join(self.figures_path,graph_name)
                    )
                plt.close()

            # appending the number of frames and exporting files 
            # with the data
            array = np.hstack(
                (np.array(frames).reshape((len(frames),1)), array))
            np.save(
                os.path.join(
                    self.results_path,'{:}.npy'.format(case_name)),
                        array
            )

        plt.close('all')
        
        

class ScalarComparator():
    """ Comparator class used to compare scalar fields.    
    """    
    def __init__(self):
        # a list indicating the stats names. If a nested list is
        # provided, each sub-listed is treated separately (generates
        # its own graph), and a group name must be provided
        self.names = (('maxSE','MSE','RMSE'),
                      ('ref','maxrSE','rMSE','rRMSE'))
        self.group_names = ('error','error_norm')
    
        # format for the filenames. It must has a place for 
        # the name of the test and the name of the scene
        self.filename_format = 'scalar_{:}_{:}'
        self._start_plot()


    def _start_plot(self):
        # starting the plot panels
        self.panel = pulsenetlib.visualize.RecursivePanelPlot(
                    200, 200, 4, 1,
                    num_rows=3,
                    colormaps=('seismic','seismic','hot'),
                    limits=(8e-4*np.array([-1,1]),
                            8e-4*np.array([-1,1]),
                            [1e-10,1]),
                    labels=('$\\mathrm{target}$\n$f(t)$',
                            '$\\mathrm{estimation}$\n$\\tilde{f}(t)$',
                             r'$\dfrac{[f(t) - \tilde{f}(t)]^2}' + \
                                  r'{\mathrm{max}(f(t))^2}$'),
                    log_colorbar=(False,False,True), 
                    cover_row=(False,True,True), 
        ) 
    

    def reset(self):
        self._start_plot()
    

    def _stats(self, field, ref_value):
        return (torch.stack((torch.max(field),
                             torch.mean(field),
                             torch.sqrt(torch.mean(field)))
                           ),
                 torch.stack((ref_value,
                              torch.mean(field)/ref_value,
                              torch.max(field)/ref_value,
                              torch.sqrt(torch.mean(field))/ref_value)
                           ) 
               )
    
    
    def _field(self, target, output):
        # square error field
        field = [torch.pow(target - output, 2.0)]
        ref_value = torch.max(target)                 
        return field, ref_value
    
    
    def __call__(self, target, output):
        field, ref_value = self._field(target, output)
        stats = self._stats(field[0], ref_value)
        # normalizing the error field for plotting
        field[0] /= ref_value**2.0              
        return field, stats


def save_VTK_2D(file_name, scalar, scalar_name, dx=1.0, dy=1.0):
    """ Function to save VTK file with a scalar
    
    The array is defined with x in rows and y in columns.
    Files can be read using pulsenetlib.preprocess.VTKLoader class.
    
    ----------
    ARGUMENTS
    
        file_name: path to the vtk file to be generated
        scalar: numpy 2D array with the scalar field
        scalar_name: string with the name
        dx, dy: grid spacing in x and y. Optional, default is 1.0
        
    ----------
    RETURNS
    
        ##none##
    
    """

    nx, ny = scalar.shape
    nz = 1
    
    import vtk
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx,ny,1)
    image_data.SetOrigin(0.0,0.0,0.0)
    image_data.SetSpacing(dx,dy,0.0)
    image_data.AllocateScalars(vtk.VTK_DOUBLE, 1);
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                image_data.SetScalarComponentFromDouble(
                    i, j, k, 0, scalar[i,j])
                
    image_data.GetPointData().GetScalars().SetName(scalar_name)
    
    vtk_writer = vtk.vtkXMLImageDataWriter()
    vtk_writer.SetFileName(file_name)    
    vtk_writer.SetInputData(image_data)
    vtk_writer.Write()