#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for importing and exporting the model, losses evolution and
configuration files.

"""

# native modules
import os
import glob
import shutil
import importlib.util

# third-party modules
import torch
import numpy as np
import yaml

# local modules
##none##


def get_file_from_regex(save_dir, regex):
    available_files = glob.glob(f'{save_dir}/{regex}')
    # sort available files
    if len(available_files) >= 1:
        available_files.sort(key=os.path.getmtime)
        filename = available_files[-1] 
    else:
        raise FileNotFoundError(
            f'No file with regex ({regex}) found in {save_dir}')
        
    return os.path.join(save_dir,filename)
    

    
def resume_configuration_files(config_dir,
             train_config_path=None,
             model_config_path=None):
    """ Function for overwriting training and model configuration dicts.
    
    Reads the yaml configuration files located in the path indicated in
    config_dir and loads the selected training and model configuration 
    dictionaries.

    ----------    
    ARGUMENTS

        config_dir: folder where the configuration files are located
        train_config_path: training configuration files filename. Optional,
                           if None, default names is used:
                           *_conf.yaml'                        
        model_config_path: model configuration files filename. Optional,
                           if None, default names is used:
                           *_mconf.yaml'

    ----------    
    RETURNS

        training_configuration: Dictionary with training configuration 
        model_configuration: Dictionary with model configuration
    
    """
    if train_config_path is None:
        regex = '*_conf.yaml'
        train_config_path = get_file_from_regex(config_dir, regex) 
    elif not os.path.isfile(train_config_path):
        raise FileNotFoundError(f'{train_config_path} does not exist')
    else:
        pass

    if model_config_path is None:
        regex = '*_mconf.yaml'
        model_config_path = get_file_from_regex(config_dir, regex) 
    elif not os.path.isfile(model_config_path):
        raise FileNotFoundError(f'{model_config_path} does not exist')
    else:
        pass
        
    # reading the yaml files and updating the dictionaries
    with open(train_config_path, 'r') as f:
        train_configuration = yaml.load(f, Loader=yaml.FullLoader)
    
    with open(model_config_path, 'r') as f:
        model_configuration = yaml.load(f, Loader=yaml.FullLoader)

    return train_configuration, model_configuration



def load_model_spec(model_dir, model_file_path=None):
    """ Function for loading a predefined model.
    
    Reads the file located in the path indicated in model_dir.
    It loads the version of the module that define the model present
    in its folder.

    ----------    
    ARGUMENTS
        model_dir: folder where the model was saved
        model_file_path: string with the name of the path to the
                         model module. Optional, if None, files with
                         the for *_model.py are searched and the most
                         recent one is loaded.

    ----------    
    RETURNS
        module_spec: ModuleSpec instance with the module defining 
                     the neural network architecture of the 
                     loaded model
    
    """
    
    # loading and copying the model script (model.py)
    if model_file_path is None:
        regex = '*_model.py'
        model_file_path = get_file_from_regex(model_dir, regex)
    
    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(
            f'At resume, {model_file_path} does not exist'
        )
    
    print(f'Loading corresponding module:\n'
          f'  {model_file_path}')
    
    module_spec = importlib.util.spec_from_file_location(
        'orig_model', model_file_path)
    
    return module_spec



def load_pl_ckpt(ckpt_dir, file_to_load=None):
    """ Function for loading a Lightning checkpoint.
    
    ----------    
    ARGUMENTS
    
        train_configuration: dictionary with the training
                             configuration
        file_to_load: string with the name of the *.pth to
                      be loaded. Optional, if None, the latest
                      *.pth file in folder is selected.

    ----------    
    RETURNS
        path_to_ckpt: /path/to/ligtning/checkpoint.ckpt 
    
    """
    
    if file_to_load is None:        
        regex = '*.ckpt'
        path_to_ckpt = get_file_from_regex(ckpt_dir, regex)
    else:
        path_to_ckpt = os.path.join(ckpt_dir, file_to_load)
    
    if not os.path.isfile(path_to_ckpt):
        raise FileNotFoundError(
            f'Checkpoint file {path_to_ckpt} not found'
       ) 
    else:
        print(f'Loading checkpoint file {path_to_ckpt}\n')
    
    return path_to_ckpt



def save_configuration_files(save_dir,
                training_configuration, model_configuration,
                train_config_path=None, model_config_path=None):
    """ Function to save yaml files with the configuration dictionaries.

    ----------
    ARGUMENTS
    
        training_configuration: training configuration dictionary
        model_configuration: model configuration dictionary
        train_config_path: training configuration files filename. Optional,
                        if None, default names is used:
                        [modelFilename]_conf.yaml'                        
        model_config_path: model configuration files filename. Optional,
                         if None, default names is used:
                         [modelFilename]_mconf.yaml'
        
    ----------
    RETURNS
    
        ##none##
    
    """    
    
    if train_config_path is None:
        train_config_path = os.path.join(
            save_dir,
            training_configuration['modelFilename'] + '_conf.yaml'
        )
            
    if model_config_path is None:
        model_config_path = os.path.join(
            save_dir,
            training_configuration['modelFilename'] + '_mconf.yaml'
        )
    
    print(f'Saving yaml files with configuration dictionaries to folder:'
          f'{save_dir}')
    
    for name, file, config in zip(
                      ['Training','Model'],
                      [train_config_path,model_config_path],
                      [training_configuration,model_configuration]):
        if os.path.isfile(file):
            print(f'  {name} configuration yaml file is being overwriten:\n'
                  f'  {file}')
            
        # create YAML files in output folder (human readable)
        with open(file, 'w') as outfile:
            yaml.dump(config, outfile)

    print('')



def save_model(model_dir, model_name=None):
    """ Function to copy the model.py file into the destination folder.

    ----------
    ARGUMENTS
    
        model_dir: path to save directory where model.py is saved
        model_name: string with the name of the file to be put on
                    the destination folder. Optional, if None,
                    _model.py is used
                  
    ----------
    RETURNS
    
        ##none##
        
    """
    
    model_dir = os.path.realpath(model_dir)

    # get pulsenet library root dir
    library_dir = os.path.join(
                    os.sep,*(
                        os.path.realpath(__file__).split(glob.os.sep)[:-1]
                        )
                    )
    
    if model_name is None:
        model_name = '_model.py'
    else:
        model_name = '_{:}_model.py'.format(model_name)
        
    destination_path = os.path.join(model_dir, model_name)
    
    # overwriting previous file, if present
    if os.path.exists(destination_path):
        print('Model file already present in folder, being overwritten...')
    origin_path = os.path.join(library_dir, 'model.py')
    shutil.copyfile(origin_path, destination_path)
            
    print(f'Copying the model.py to the destination folder:\n' 
          f'  {origin_path}\n'
          f'   to\n'
          f'  {destination_path}\n')



def load_model(model_dir, ckpt_format='./checkpoints/epoch=*.ckpt',
               run_device=None):
    """ Load model and state from files in folder
    
    TODO: merge this with the load_checkpoint function, since they
    are pretty much the same function!

    ----------    
    ARGUMENTS
        
        model_dir: path to the model training files
        ckpt_format: relative path (from the models) to the checkpoint file
                     Used to select a given checkpoint
        run_device: running device. Optional, default is to check and 
                select GPU if available.
    
    ----------    
    RETURNS
        
        net: neural network, updated to the checkpoint on ckpt_path
        state_dict: model's state dict
        train_configuration: dictionary with the training configuration
        model_configuration: dictionary with the model configuration
    
    """

    try:
        model_file_path, config_path, model_config_path, path_to_ckpt = \
            get_default_files(model_dir, ckpt_format=ckpt_format)
        print('Selected checkpoint:\n {:}\n'.format(path_to_ckpt))
        # boolean indicating must consider legacy file formats and functions
        is_legacy = False
    except FileNotFoundError:
        model_file_path, config_path, model_config_path, path_to_ckpt = \
            get_default_files(model_dir, ckpt_format='*epoch*.pth')
        is_legacy = True

    if run_device is None:
        if torch.cuda.is_available():
            run_device = torch.device('cuda:0')
        else:
            run_device = torch.device('cpu')
    
    checkpoint = torch.load(
        path_to_ckpt, map_location=run_device)
   
    module_spec = importlib.util.spec_from_file_location(
        'orig_model', model_file_path)
    loaded_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loaded_module)
    
    # reading the yaml files of the configuration dictionaries
    train_configuration = {}
    model_configuration = {}
    with open(config_path, 'r') as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
        train_configuration.update(temp)    
    with open(model_config_path, 'r') as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
        model_configuration.update(temp)
    
    # model properties
    channels = model_configuration['channels']
    number_channels = len(channels)
    input_frames = model_configuration['numInputFrames']
    output_frames = model_configuration['numOutputFrames']
    std_norm = torch.tensor(
        model_configuration['stdNorm']).to(device=run_device)
    avg_remove = torch.tensor(
        model_configuration['avgRemove']).to(device=run_device)
    
    # network
    net = loaded_module.PulseNet(number_channels,
                                 input_frames,
                                 output_frames,
                                 std_norm,
                                 avg_remove)
    
    if is_legacy:
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
    else:
        # create a stat_dict with renamed keys (only network, not model)
        state_dict = dict()
        for key in checkpoint['state_dict'].keys():
            state_dict[key.replace('model.','')] = \
                checkpoint['state_dict'][key]
        net.load_state_dict(state_dict)

    return net, state_dict, train_configuration, model_configuration



def get_default_files(model_dir,
                      ckpt_format='./checkpoints/epoch=*.ckpt'):
    """ For a given folder, returns the paths to model training files.
    
    Its is based on the files default names and locations.
    
    ----------
    ARGUMENTS
    
        model_dir: path to folder
        ckpt_format: path with the format of the state file.
                     Optional, default is './checkpoints/epoch=*.ckpt'
    
    ----------    
    RETURNS
    
        model_path: path of the model .py script
        config_path: path of the training configuration .yaml file
        mconfig_path: path of the model configuration .yaml file
        ckpt_path: path of the model configuration .ckpt file. If more
                   than one file is available on folder, the one that
                   was modified last is selected
    
    """
    paths = []
    for regex in ['*_model.py', '*_conf.yaml',
                  '*_mconf.yaml', ckpt_format]:
        paths.append(
            get_file(os.path.join(model_dir, regex))
            )
    model_path, config_path, mconfig_path, ckpt_path = paths

    return model_path, config_path, mconfig_path, ckpt_path



def get_file(file_path):
    """ Return full path to a regex path.
    If multiple exist, return the youngest one.
    """
    files = glob.glob(file_path)
    if len(files) == 0: raise FileNotFoundError(f'{file_path} not found.')
    elif len(files) > 1: files.sort(key=os.path.getmtime)
    return files[-1]



def LEGACY_save_checkpoint(state, save_path, file_name, epoch, is_best=False):
    """ Function to save the model state in a *.pth file using torch.save(...)
    ----------
    ARGUMENTS

        state: the model to be saved
        save_path: path where to save the file
        file_name: string with the filename name including a format
                  for the epoch number
        epoch: epoch number to be added to filename
        is_best: boolean indicating if it is the best model until
                 now. If true, previous best in folder is removed and
                 a copy of the current model file with the suffix best
                 is added to folder.
                 
    ----------
    RETURNS

        ##none##

    """
    
    file_path = glob.os.path.join(
        save_path, file_name.format(epoch) + '.pth')
    
    torch.save(state, file_path)

    # copy if it is the best result until now
    if is_best:
        base_name = \
            file_name.split('{')[0] + file_name.split('}')[1]
    
        # remove epoch number indication
        previous_best_path = \
            glob.glob(
                glob.os.path.join(
                    save_path, base_name + 'Best_*.pth')
                     )

        # remove previous best if exists
        if previous_best_path:
            try:
                os.remove(previous_best_path[0])
            except FileNotFoundError:
                pass

        new_best_path = \
            glob.os.path.join(
                save_path, base_name + 'Best_{:04d}.pth'.format(epoch)
                             )
        
        # make a copy of the new best (recently saved)
        shutil.copyfile(file_path, new_best_path)    



def LEGACY_load_configuration(train_configuration, model_configuration):
    print('Overwriting conf and file_mconf')
    config_path = glob.os.path.join(
        train_configuration['modelDir'],
        train_configuration['modelFilename'] + '_conf.yaml')
    
    model_config_path = glob.os.path.join(
        train_configuration['modelDir'],
        train_configuration['modelFilename'] + '_mconf.yaml')
    
    assert glob.os.path.isfile(config_path), \
        config_path + ' does not exist!'
    assert glob.os.path.isfile(model_config_path), \
        model_config_path + ' does not exist!'
        
    # reading the yaml files and updating the dictionaries
    with open(config_path, 'r') as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
        train_configuration.update(temp)
    
    with open(model_config_path, 'r') as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
        model_configuration.update(temp)

    return train_configuration, model_configuration



def LEGACY_load_checkpoint(train_configuration, model_configuration,
                    file_to_load=None,
                    model_file_path=None):
    """ Function for loading a predefined model.
    
    Reads the file defined located in the path indicated in the
    conf dictionary (modelDir, modelFilename). It loads the version of
    the module that define the model present in its folder.
    ----------    
    ARGUMENTS
    
        train_configuration: dictionary with the training
                             configuration
        model_configuration: dictionary with the model
                             parameters 
        file_to_load: string with the name of the *.pth to
                      be loaded. Optional, if None, the latest
                      *.pth file in folder is selected.
        model_file_path: string with the name of the path to the
                         model module. Optional, if None,
                         _[modelFilename]_model.py is used.
    ----------    
    RETURNS
    
        train_configuration: updated dictionary
        model_configuration: updated dictionary
        state: loaded model state
        module_spec: ModuleSpec instance with the module defining 
                     the neural network architecture of the 
                     loaded model
    
    """
    
    print('Loading checkpoint')
       
    if file_to_load is None:        
        print('Searching newest *.pth file in folder')
        available_files = glob.glob(
            '{:}/{:}*.pth'.format(
                train_configuration['modelDir'],
                train_configuration['modelFilename'])
            )
        
        # sort available files
        available_files = sorted(
            available_files, key=os.path.getmtime)
        # selecting the last one
        model_path = available_files[-1]
        
    else:
        model_path = glob.os.path.join(
            train_configuration['modelDir'],
            train_configuration['modelFilename'] + file_to_load)
        
    assert glob.os.path.isfile(model_path), \
        model_path + ' does not exist!'
    
    # loading model state
    print('Loading:\n  {:}'.format(model_path))
    state = torch.load(model_path)

    train_configration, model_configuration = \
         load_configuration(train_configuration, model_configuration)
    
    model_dir = os.path.realpath(train_configuration['modelDir'])
    
    # loading and copying the model script (model.py)
    if model_file_path is None:
        model_file_path = glob.os.path.join(
            model_dir,'_{:}_model.py'.format(
                train_configuration['modelFilename']))
    
    assert glob.os.path.isfile(model_file_path), \
        model_file_path + ' does not exist!'
    
    print('Copying and loading corresponding model module:')
    print('  {:}'.format(model_file_path))
    
    module_spec = importlib.util.spec_from_file_location(
        'orig_model', model_file_path)
    
    return train_configuration, model_configuration, \
           state, module_spec
    


def LEGACY_export_losses(file_name, data_new):
    """ Function to save *.npy files with the losses evolution.
    ----------
    ARGUMENTS
    
        file_name: string with the path for loading/saving data
        data_new: numpy array with the data to be concatenated
        
    ----------    
    RETURNS
    
        ##none##
    
    """
    
    # allocate array
    data_old = np.empty((0, data_new.shape[1]),
                        dtype=float)
    
    # read previous loss file in folder
    # if exists, concatenate the result with the new values
    if glob.os.path.isfile(file_name):
        data_old = np.load(file_name, allow_pickle=True)
    
    data = np.append(
        data_old, data_new, axis=0)
    
    # save file
    np.save(file_name, data)



def LEGACY_save_config(training_configuration, model_configuration,
                file_conf_yaml=None, file_mconf_yaml=None):
    """ Function to save yaml files with the configuration dictionaries.
    
    If already there, files are not overwritten.
    ----------
    ARGUMENTS
    
        training_configuration: training configuration dictionary
        model_configuration: model configuration dictionary
        file_conf_yaml: training configuration files filename. Optional,
                        if None, default names is used:
                        [modelFilename]_conf.yaml'                        
        file_mconf_yaml: model configuration files filename. Optional,
                         if None, default names is used:
                         [modelFilename]_mconf.yaml'
        
    ----------
    RETURNS
    
        ##none##
    
    """    
    
    model_dir = os.path.realpath(
        training_configuration['modelDir'])
    
    if file_conf_yaml is None:
        file_conf_yaml = glob.os.path.join(
            model_dir,'{:}_conf.yaml'.format(
                training_configuration['modelFilename'])
        )
            
    if file_mconf_yaml is None:
        file_mconf_yaml = glob.os.path.join(
            model_dir,'{:}_mconf.yaml'.format(
                training_configuration['modelFilename'])
        )
    
    print(f'Saving yaml files with configuration dictionaries to folder:'
          f'{model_dir}')
    
    for name, file, config in zip(
                      ['Training','Model'],
                      [file_conf_yaml,file_mconf_yaml],
                      [training_configuration,model_configuration]):
        if os.path.isfile(file):
            print('  {:}: configuration yaml file is being overwriten'.format(
                name))
        print('  {:}'.format(file))
            
        # create YAML files in output folder (human readable)
        with open(file, 'w') as outfile:
            yaml.dump(config, outfile)

    print('')



def LEGACY_copy_model(model_dir, model_name=None):
    """ Function to copy the model.py file into the destination folder.
    ----------
    ARGUMENTS
    
        model_dir: path to save the copy of model.py
        model_name: string with the name of the file to be put on
                    the destination folder. Optional, if None,
                    _model.py is used
                  
    ----------
    RETURNS
    
        ##none##
        
    """
    
    model_dir = os.path.realpath(model_dir)
    
    library_dir = glob.os.path.join(
        glob.os.sep,*(
            os.path.realpath(__file__).split(glob.os.sep)[:-1]))
    
    if model_name is None:
        model_name = '_model.py'
    else:
        model_name = '_{:}_model.py'.format(model_name)
        
    copy_model_path = glob.os.path.join(model_dir, model_name)
    
    # overwriting previous file, if present
    if glob.os.path.exists(copy_model_path):
        print('Model file already present in folder, being overwritten...')
    shutil.copyfile(library_dir + '/model.py', copy_model_path)
            
    print('Copying the model.py to the destination folder:')    
    print('  {:}'.format(library_dir + '/model.py'))
    print('     to')
    print('  {:}\n'.format(copy_model_path))



if __name__ == "__main___":
    pass