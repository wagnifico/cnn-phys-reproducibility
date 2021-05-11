#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Sample code for the testing.
This script may take a lot of RAM due to the use of recursion and
multiple images generation.

To run, a configuration file is demanded:
python test.py --test_config param_test.xml

if calling from a different folder, include
'sys.path.append(path/to/pulsenetlib)' before importing the
module.


"""

# native modules
import sys
import argparse

# third party modules
import yaml
import torch
import numpy as np

# local modules
import pulsenetlib


class SampleMultipleComparator():
    """ Sample of the Comparator class used to test the model.
    
    It must take the target and the output fields only and return
    error fields (as a list) to be plotted and a stats variable
    that contains statistics of the given fields.

    The stats may be organized as a nested list - such as
    stats = ((a1,a2,a3),(b1,b2,b3)) - so the quantities are plotted   
    and stored separately (e.g., one plot/file with a1, a2 and a3 and
    the other with b1, b2 and b3).

    This sample case calculates the square-error and its statistics
    (mean, root-mean), raw values and also normalized by the maximun
    value at the target frame. The square-error of the divergence
    field is also calculated and plotted, with normalized stats being 
    calculated as well.
    
    """
    
    def __init__(self):
        # a list indicating the stats names. If a nested list is
        # provided, each sub-listed is treated separately (generates
        # its own graph), and a group name must be provided
        self.names = (('maxSE','MSE','RMSE'),
                      ('ref','maxrSE','rMSE','rRMSE'),
                      ('div_ref','div_maxrSE','div_rMSE','div_rRMSE'))
        self.group_names = ('error','error_norm','div_error_norm')
    
        # format for the filenames. It must has a place for 
        # the name of the test and the name of the scene
        self.filename_format = 'scalar_div_{:}_{:}'
        self._start_plot()


    def _start_plot(self):
        # starting the plot panels
        self.panel = pulsenetlib.visualize.RecursivePanelPlot(
                    200, 200, 4, 1,
                    num_rows=4,
                    colormaps=('seismic','seismic','hot','viridis'),
                    limits=(8e-4*np.array([-1,1]),
                            8e-4*np.array([-1,1]),
                            [1e-10,1],
                            [1e-10,1]),
                    labels=('$\\mathrm{target}$\n$f(t)$',
                            '$\\mathrm{estimation}$\n$\\tilde{f}(t)$',
                            r'$\dfrac{[f(t) - \tilde{f}(t)]^2}' + \
                                 r'{\mathrm{max}(f(t)^2)}$',
                            r'$\dfrac{[\nabla f(t) - \nabla \tilde{f}(t)]^2}' + \
                                 r'{\mathrm{max}(\nabla f(t)^2)}$'),
                    log_colorbar=(False,False,True,True), 
                    cover_row=(False,True,True,True), 
        )


    def reset(self):
        self._start_plot()         


    def _stats(self, fields, ref_values):
        return (torch.stack((torch.max(fields[0]),
                             torch.mean(fields[0]),
                             torch.sqrt(torch.mean(fields[0])))
                           ),
                 torch.stack((ref_values[0],
                              torch.mean(fields[0])/ref_values[0]**2,
                              torch.max(fields[0])/ref_values[0]**2,
                              torch.sqrt(torch.mean(fields[0]))/ref_values[0])
                           ),
                 torch.stack((ref_values[1],
                              torch.mean(fields[1])/ref_values[1]**2,
                              torch.max(fields[1])/ref_values[1]**2,
                              torch.sqrt(torch.mean(fields[1]))/ref_values[1])
                           )
               )
    
    
    def _field(self, target, output):
        fields, ref_values = [], []
        
        # scalar error field
        fields.append(torch.pow(target - output, 2.0))
        ref_values.append(target.abs().max())         
        
        # divergence error field
        dx = 100/(200 - 1)
        div_target = pulsenetlib.tools.divergence(target,dx)
        div_output = pulsenetlib.tools.divergence(output,dx)
        fields.append(torch.pow(div_target - div_output, 2.0))
        ref_values.append(div_target.abs().max())    
        
        return fields, ref_values
    
    
    def __call__(self, target, output):
        fields, ref_values = self._field(target, output)
        stats = self._stats(fields, ref_values)
        
        # normalizing the error field for plotting
        fields[0] /= ref_values[0]**2.0   
        fields[1] /= ref_values[1]**2.0   
                 
        return fields, stats

        

if __name__ == '__main__':

    # current code is not GPU compatible
    print('\nNot using GPU.')
    torch.set_default_tensor_type(torch.DoubleTensor)
    run_device = torch.device('cpu')
    
    # no defaults (None), so if passed replaces the file values
    parser = argparse.ArgumentParser(description='Testing script')

    parser.add_argument('--test_config',
                        help='Path to the *.yaml file with testing parameters.\n' + 
                             'If flags are given, they overwrite the values on file.')
    parser.add_argument('--model_path',
                        help='Path to the model')
    parser.add_argument('--ckpt_epoch', default=None, type=int,
                        help='Epoch number of the checkpoint to be used. Default behavior is to consider the highest epoch.')                    
    parser.add_argument('--reference_path',
                        help='Path to the reference cases. Must be a folder with ' +
                             'simulation subfolders.')
    parser.add_argument('--file_format',
                        help='Format reference file to be read.')
    parser.add_argument('--comparison_name',
                        help='Comparison name, to be appended to output files')
    parser.add_argument('--frame_step', type=int,
                        help='Number of solution frames to jump when testing model')
    parser.add_argument('--channel_index', type=int,
                        help='Index to the channel to be compared (default is 0)')
    parser.add_argument('--frame_start', type=int,
                        help='Index of the starting frame (number of timesteps)')
    parser.add_argument('--frames_to_estimate', type=int,
                        help='Number of frames to evaluate (recurrencies + 1).\n'+
                             'Default is 0, all possible frames are analyzed.')
    parser.add_argument('--no_panel', action='store_true', default=False,
                        help='Not generating the comparison panel.')
    parser.add_argument('--no_graphs', action='store_true', default=False,
                        help='Not generating the errors evolution graphs.')   
    parser.add_argument('--export_field', action='store_true', default=False,
                        help='Generating VTK files with estimation.') 
    parser.add_argument('--apply_EPC', action='store_true', default=False,
                        help='Applying energy correction in the recurrent analysis.')
    parser.add_argument('--number_workers', type=int,
                        help='Number of processes used to perform the test.\n' +
                             'Optional, default is 12.')
    parser.add_argument('--verbose', action='store_true',
                        help='')       

    parsed_args = parser.parse_args()
        
    # reading parameters file, overwriting values with passed inputs
    # and defining default values
    with open(parsed_args.test_config, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    for arg_key in vars(parsed_args):
        value = getattr(parsed_args, arg_key)
        if value is not None:
            args[arg_key] = value
    
    replace_default = lambda key, default: \
        args.update({key:default}) if (
            key not in args or args[key] is None) else 0
    
    replace_default('comparator', SampleMultipleComparator())
    replace_default('verbose', False)
    replace_default('number_workers', 12)    
    
    generate_sub_dict = lambda dict, keys : {
        key:dict[key] for key in keys if key in args}    

    tester_args = generate_sub_dict(
        args, ('model_path','reference_path','comparator',
               'file_format','comparison_name', 'apply_EPC'))
    tester_args['run_device'] = run_device

    ckpt_epoch = getattr(parsed_args,'ckpt_epoch')
    if ckpt_epoch is None: ckpt_epoch = '*'
    tester_args['checkpoint_file_path'] = \
        f'./checkpoints/epoch={ckpt_epoch}.ckpt'
    
    args['plot_panel'] = not args['no_panel']
    args['plot_graphs'] = not args['no_graphs']
    args['export_field'] = args['export_field']
    run_args = generate_sub_dict(
        args, ('frame_start','frames_to_estimate',
               'plot_panel','plot_graphs','export_field',
               'number_workers','verbose'))
   
    print(
        '\nConsidered folders:\n|  Model:  {:}\n|  Reference: {:}\n'.format(
            args['model_path'], args['reference_path'])
    )
    
    if args['tester'] is None:
        tester = pulsenetlib.tester.RecursiveTester(
                     **tester_args)
    
    tester.run(**run_args)

