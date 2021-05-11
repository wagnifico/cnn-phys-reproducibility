#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating and defining the database
"""


# native modules
import sys
import os
import glob
import errno
import functools
import re
from typing import Callable, Optional, Sequence, Tuple

# third-party modules
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.multiprocessing
import pytorch_lightning as pl

# local modules
from pulsenetlib.utilities import calltracker 


class PulseNetDataset(Dataset):
    """ Pulse Net dataset loader.
    
    This groups data in groups of (N+1) consecutive timesteps (from the
    same simulation, i.e. the same folder). It considers that all the folders
    inside the given dataset folder for storing simulation results and that
    all those folders have the same amount of solution steps.
    
    For reading both raw and pre-processed data, a 1D list of paths
    for each category of files is generated during the initiation of the
    class. Thus, a single integer is used as input for reading the data.
    Each processed file consists of a sample (input + target frames),
    for the considered frame step.

    # TODO: Is this better than data augmentation at training time?
    For performance, the data augmentation operations are performed
    on preprocessing. One must be aware that the processed data may be
    different from raw data.
    
    A PulseNetDataset is a torch map-style dataset and implements 3 key methods:
    
    * **setup(stage)** (must be called after object instantiaton `stage` = ('preprocess', 'run')).
    * **__len__()** (dataset length) 
    * **__get_item__()** (loads one dataset item)
   
    # TODO: put the list of arguments in the same order as they are used!
    ARG:        
        path_raw_format: string with the path to the raw data. Includes the
                         format (in python formatting) for the numerical
                         references of the scenes and frames,
                         e.g './data/training/{:04d}/data_{:02d}.vti'
        path_processed_format: string with the path to the processed data.
                               Generated files will contain all the
                               frames used on the training. Includes the
                               format (in python formatting) for the numerical
                               references of the scenes and group
                               (as in previous variable)
        frames: tuple with the indications of number of frames and frame
                step to consider for the generation of the database:
                  - number of input frames;
                  - number of target frames;
                  - frame step (in number of simulation physical timesteps)  
        channels: list of strings indicating the name of the channels
                  (fields) that will be used considered
                  TODO: maybe remove this in the future, since information
                  is already contained in rawloader
        rawloader: function that reads the raw files. Must only take the 
                  file path as argument and returns a tensor
        max_data_points: integer indicating the maximum number of data points
                         to consider on pre-processing. In order to favor
                         diversity, splits the number of data points in
                         the scenes rather than selecting a reduced number
                         of full scenes. If 0, all the data
                         available on the raw database folder will be used.
                         Optional, default is 0.
        label: string indicating the type of database, added to configuration
               file. Normally, 'training' or 'validation'. Optional,
               default is ''.
        data_augmentation: function to apply when loading data to perform 
                           image augmentation. The data augmentation is
                           Example: flip images randomly. Optional, 
                           default is none
        preprocess: set to True for preprocessing the dataset and exiting
                    execution after. Optional, default is False
        num_preproc_threads: number of threads to perform parallel
                             preprocessing of dataset. Optional, default is 1
        select_mode: string that defines how to select the groups of frames
                     in scene, more info in method _set_select_mode.
                     Optional, default is 'first'
    
    """   

    def __init__(self,
        path_processed_format: str,
        frames: Tuple[int, int, int],
        channels: Sequence[str],
        max_data_points: Optional[int] = 0,
        label: Optional[str] = '',
        data_augmentation: Optional[Callable] = None,
        preprocess: Optional[bool] = False,
        path_raw_format: Optional[str] = None,
        rawloader: Optional[Callable[[str], Tuple[torch.Tensor, int, int]]] = None, 
        num_preproc_threads: Optional[int] = 1,
        select_mode: Optional[str] = 'first',
    ):
        if preprocess:
            if path_raw_format is None:
                raise ValueError(
                        'path_raw_format = None for preprocess=True'
                    )
            if rawloader is None:
                raise ValueError(
                        'rawloader = None for preprocess=True'
                    )

        self.num_input_frames = frames[0]
        self.num_target_frames = frames[1]
        self.frame_step = frames[2]
        self.channels = channels
        self.num_channels = len(channels)       
        self.max_data_points = max_data_points        
        self.data_augmentation = data_augmentation

        self.n_threads = num_preproc_threads 
        
        # number of consecutive frames that are captured to form
        # each group
        self.frames_per_group = self.num_input_frames + self.num_target_frames
        
        self._set_select_mode(select_mode)

        # Log file (exists if dataset was preprocessed in the past).
        # It contains a dict with the name of data tensors and
        # target tensors
        self.preprocess_log = {}

        if preprocess:
            self.setup('preprocess', label,
                       path_processed_format, path_raw_format, rawloader)
        else:
            self.setup('run', label, path_processed_format)


    
    def _set_select_mode(self, select_mode):
        """ Setting how the groups in the scene are going to be selected

        Attribute select_fun, a callable that will return a subset
        of a given array following the selection mode selected by 
        select_mode, is added to the instance.
        
        Only used when the number of selected groups is smaller than
        the total number of available groups (maxDataPoints different
        than 0). 
        
        ----------
        ARGUMENTS

            select_mode: string defining the selection mode. Options:
                         - 'first': first n groups (start of the scene)
                         - 'last': last n groups (end of scene)
                         - 'random': random n groups, uniform distribution
 
        ----------
        RETURNS

            ##none##
        
        """
        
        select_names = ['first','last','random']
        select_funs = [
            self._select_first, self._select_last, self._select_random]
        which_select_mode = [select_mode == mode for mode in select_names]   
        
        if np.any(which_select_mode):
            index_mode = np.arange(
                len(select_names),dtype=int)[which_select_mode][0]
            self.select_fun = select_funs[index_mode]
        else:
            raise ValueError(
                'Select mode does not correspond to any of the '
                'implemented (first, last or random)')
   
    def _select_first(self, arr, n):
        return arr[:n]

    def _select_last(self, arr, n):
        return arr[-n:]

    def _select_random(self, arr, n):
        return arr[np.random.choice(len(arr), n, replace=False)]



    @calltracker
    def setup(self, stage, label, path_processed_format,
              path_raw_format=None, rawloader=None):
        """
            Processes the dataset.
            Must be called before loading any data (using __get_item__())
        """

        # create pre-processed data folder, if not already there
        self.path_data_dir, _, _ = list_available_files(path_processed_format)
        if (not os.path.exists(self.path_data_dir)):
            os.makedirs(self.path_data_dir)
        self.__check_folder__(self.path_data_dir)
        log_path = os.path.join(
            self.path_data_dir, f'preprocessed_{label}.yaml')
        
        if stage == 'preprocess':
            self._preprocess_params(path_raw_format, path_processed_format)
            self._preprocess(log_path, rawloader)
        elif stage == 'run':
            self._process_params(path_processed_format)
            self._process(log_path)
            self._check_preproc_parameters(log_path)
        else:
            raise ValueError(
                'In setup, stage arg must be either `preprocess` or `run`.')



    def _preprocess_params(self, path_raw_format, path_processed_format):
            # get all the available files that respect the raw
            # file format in folder
            path_raw_dir, list_raw_files, references_raw_files = \
                    list_available_files(path_raw_format)
            
            # check if folder exists
            self.__check_folder__(path_raw_dir)
            
            if self.data_augmentation is not None:
                print(
                    f'\nWARNING: Data augmentation operations will '
                    f'be performed on the pre-processing!\n'
                    f'Be aware that the pre-processed dataset '
                    f'may be different from raw data.\n'
                )
                
            # check number of scenes
            self.number_scenes = np.shape(list_raw_files)[0]
            if self.number_scenes <= 0:
                raise ValueError(f'No scenes found in {self.path_raw_dir}')
            
            print('\nChecking consistency of the database...\n|')
        
            # check on the scenes (folders) a minimal common number of
            # steps (files). For a more improved version of the code,
            # ignore this condition and consider as much as possible
            # for each simulation        
            steps_per_scene = float('inf')

            for scene_index, scene_files in enumerate(list_raw_files):
                # mask for selecting the files of a given scene
                number_available_frames = len(scene_files)
                
                if number_available_frames == 0:
                    print(
                        f'|  Scene {scene_index:d} '
                        f'has zero steps! It is ignored.'
                    )

                    # remove from list of paths and folders
                    list_raw_files.remove(scene_files)        
                else:
                    # get the lower number of available steps
                    steps_per_scene = min(
                        number_available_frames, steps_per_scene)
            
            # preparing the list of files such as to account for the 
            # framestep (can be different from the step used in the 
            # generation of the database)
            available_time_step = np.diff(references_raw_files[:2,1])
            
            if available_time_step == self.frame_step:
                pass
                # do nothing
            elif self.frame_step % available_time_step == 0:
                # selecting the multiples of the available files
                temp_list_raw_files = []
                jump_step = int(self.frame_step // available_time_step)
                print(f'|  considering 1 every {jump_step:d} files')
                for scene_files in list_raw_files:
                    temp_list_raw_files.append(scene_files[::jump_step])
                    steps_per_scene = min(
                            len(temp_list_raw_files[-1]), steps_per_scene)
                    
                list_raw_files = temp_list_raw_files
                del temp_list_raw_files
            else:
                msg =  (
                    f'The selected frame step {self.frame_step} is incompatible '
                    f'with the available simulation step.'
                )
                raise ValueError(msg) 
            # number of scenes and steps to pre-process
            self.groups_per_scene = \
                steps_per_scene // self.frames_per_group
 
            number_data_points = self.__len__()
            
            if self.max_data_points > 0 and self.max_data_points < number_data_points:
                print(
                    f'|  max number of datapoints ({self.max_data_points:d}) '
                    f'is smaller than the number '
                    f'of available cases ({number_data_points:d})'
                )

                # recalculating the number of groups per scene such as to
                # achieve the number of selected cases
                new_groups_per_scene = self.max_data_points // self.number_scenes
                
                # limiting the number of scenes to achieve the desired
                # number of datapoints             
                if new_groups_per_scene == 0:
                    new_groups_per_scene = 1
                    self.number_scenes = self.max_data_points
                    list_raw_files = list_raw_files[:self.number_scenes]
                    print(f'|  ATTENTION: number of datapoins is too small, '
                          f'a sample of the available scenes is considered.')
                
                new_steps_per_scene = new_groups_per_scene * self.frames_per_group
                
                # update number of groups per scene
                self.groups_per_scene = new_steps_per_scene // self.frames_per_group
                print(f'|  considering {new_steps_per_scene:d} frames per scene '
                      f'instead of {steps_per_scene:d}')
                print(f'|  new number of datapoints: {self.__len__()}')
                
                steps_per_scene = new_steps_per_scene      
            
            self._print_statistics(path_raw_dir)
                     
            self.raw_files = []
            for scene_files in list_raw_files:
                # list of last frames for possible non-overlaping groups of frames
                possible_ends = np.arange(start=self.frames_per_group,
                                          stop=len(scene_files)+1,
                                          step=self.frames_per_group)
                selected_ends = self.select_fun(possible_ends, self.groups_per_scene)
                for end_frame in selected_ends:
                    self.raw_files.append(
                        scene_files[end_frame-self.frames_per_group:end_frame]
                    ) 
            
            # generating a list of the preprocessed files paths. In the case of
            # random group selection, the group numbering has no relationship with
            # physical time they represent
            self.processed_files = []
            for scene_index in range(self.number_scenes):
                for group_index in range(self.groups_per_scene):
                    self.processed_files.append(
                        path_processed_format.format(scene_index, group_index)
                    )
           


    def _preprocess(self, log_path, rawloader):
        if (os.path.isfile(log_path)):
            print(
                f'For dataset in:  {self.path_data_dir} '
                f'a log file exists showing a preprocessing in the past.\n'
            )
        
            with open(log_path) as f:
                temp = yaml.load(f, Loader=yaml.FullLoader)
                self.preprocess_log.update(temp)

            # printing the log
            print('Previous log file:')
            [print('  ',key,':',self.preprocess_log[key])
               for key in self.preprocess_log]
            print('\nWARNING: This will overwrite previous ' +
                  'pre-processed files and take some time.')
        
            self.nx = self.preprocess_log['nx']
            self.ny = self.preprocess_log['ny']
        
        else:
            print(
                f'No log file found in: {self.path_data_dir}\n'
                f'Preprocessing automatically.\n'
            )
               
            # read a sample of the database to define the array size
            # no previous check on the consistency of the remaining
            # files is performed
            _, self.nx, self.ny = rawloader(self.raw_files[0][0])
        
        # perform preprocessing
        self._exec_preprocess(log_path, rawloader)


    def _process_params(self, path_data_format):
        # get all the available files that respect the raw
        # file format in folder
        _, list_processed_files, _  = list_available_files(path_data_format)
        
        self.number_scenes = len(list_processed_files)
        self.groups_per_scene = len(list_processed_files[0])
        self.processed_files = list_processed_files
        
        # convert to single dimension (number scene x number group)
        self.processed_files = np.reshape(
            self.processed_files, (self.processed_files.size))
        
        print('\nChecking consistency of the database...\n|')
        self._print_statistics(self.path_data_dir)


    def _process(self, log_path):
        # check if log file is there, raise error if not
        if not (glob.os.path.isfile(log_path)):
            raise ValueError(
                f'No log file found in {log_path}, ' + 
                 'please create one by preprocessing the dataset.'
            )
        
        with open(log_path) as f:
            temp_dict = yaml.load(f, Loader=yaml.FullLoader)

        # check if the pre-processed fields are the ones
        # to be used on current training, raising error
        # if any is missing/different
        #fields = temp_dict['fields']
        selected_channels_str = ', '.join(self.channels)
        available_fields_str = ', '.join(temp_dict['fields'])
        channel_error_msg = (
            f'Requested channels ({selected_channels_str}) is different '
            f'from the available preprocessed channels ({available_fields_str}).\n'
            f'Change input channels or re-perform preprocessing!'
        )
        
        if len(temp_dict['fields']) != self.num_channels:
            raise ValueError(channel_error_msg)
        
        # since order must be the same, comparing them directly
        for available_channel, requested_channel in zip(
            temp_dict['fields'], self.channels):
                if not available_channel == requested_channel:
                    raise ValueError(channel_error_msg)

        temp_num_input_frames = temp_dict['NumInputFrames']
        temp_num_target_frames = temp_dict['NumMaxFutureFrames']
        
        for step_name, num_frames, temp_num_frames in zip(
                ('input','target'),
                (self.num_input_frames,self.num_target_frames),
                (temp_num_input_frames,temp_num_target_frames)
            ):
            if num_frames != temp_num_frames:
                msg = (
                    f'Requested {step_name} frames ({num_frames}) is different '
                    f'from the available preprocessed {step_name} frames ({temp_num_frames}).\n'
                    f'Change input frame number or re-perform preprocessing!'
                )
                raise ValueError(msg)       

        # update preprocess dict with the loaded values
        self.preprocess_log.update(temp_dict)


    def _check_preproc_parameters(self, data_path):
        # call error for insufficient number of steps in scene
        if self.groups_per_scene < 1:
            msg = (
                f'Dataset cannot be preprocessed with current conf values.\n'
                f'The number of grouped frames must be greater than 0.\n'
                f'Found values are:\n'
                f'  group per scene: {self.groups_per_scene}\n'
                f'  conf number of input frames: {self.num_input_frames}\n'
                f'  conf max number of future frames: {self.num_target_frames}\n'
                f'  data path: {data_path}\n'
            )
            raise ValueError(msg)


    def _print_statistics(self, save_dir):
        print(f"|  Dataset: {save_dir}:\n"
              f"|  selected channels: ({', '.join(self.channels)})\n"
              f"|  number of scenes: {self.number_scenes}\n"
              f"|  number of groups per scene: {self.groups_per_scene}\n"
              f"|  total number of samples (scenes x groups): {self.__len__()}\n"
              f"|\n"
              f"|done\n"
            )


    def _exec_preprocess(self, log_path, rawloader):
        """ Preprocess the dataset from raw files to .pt (much faster I/O)
        
        Considers the files listed on attributes 'raw_files' and
        'processed_files' for, respectively, loading and exporting.
        
        """
        
        # create parent folder if it does not exists
        for i in range(self.number_scenes):
            folder = glob.os.path.dirname(
                self.processed_files[i*self.groups_per_scene])
            
            if (not glob.os.path.exists(folder)):
                print(f'Creating folder:\n{folder}\n')
                glob.os.makedirs(folder)
                
        # shared tensor to count the number of saved files
        counter = torch.tensor(0,dtype=torch.int16)
        counter.share_memory_()
        data_inputs = []
        for i in range(len(self.raw_files)):
            data_inputs.append([i, counter])
               
        print('Pre-processing dataset...\n')
        
        # create pool of works for the pre-processing
        self.dtype = torch.tensor([0.0]).dtype
        torch.multiprocessing.set_start_method('spawn', force=True)
        p = torch.multiprocessing.Pool(self.n_threads)
        func = functools.partial(
            self.__getitempreprocess__, loader=rawloader)
        # mapping the function to the selected number of workers
        p.map(func, data_inputs)
       
        print('\nPre-processing succeeded')
        
        self.preprocess_log = {
                'Directory' : os.path.dirname(folder),
                'NumInputFrames' : self.num_input_frames,
                'NumMaxFutureFrames' : self.num_target_frames,
                'NumberOfDataPoints' : self.__len__(),
                'nx' : self.nx,
                'ny' : self.ny,
                'fields': self.channels
            }
        
        print('Log is now:')
        [print('  ',key,':',self.preprocess_log[key])
           for key in self.preprocess_log]
        
        with open(log_path, 'w') as f:
            yaml.dump(self.preprocess_log, f)


    def __getitempreprocess__(self, inputs, loader):
        """ Method to load raw data and save in the desired format.
        
        Exports a single tensor in format (N, C, D, H, W). Each channel
        is a physical quantity and the dimensions indicates the different
        time frames.
        
        Performs the data augmentation operations, if any.
        
        ----------
        ARGUMENTS
        
            inputs: a list containing the inputs, they are:
                    - idx: sample index, that defines both the raw
                           and processed file's path
                    - counter: shared torch variable used to count
                               the number of preprocessed files

        ----------
        RETURNS
        
            ##none##
        
        """
                
        idx, counter = inputs
        if self.dtype == torch.float64:
            # since it is an spawned cuda process, forcing the desired
            # floating point precision
            torch.set_default_tensor_type(torch.DoubleTensor)

        # allocate tensor (input + target) to be saved
        data = torch.zeros(
            self.num_channels, self.frames_per_group, self.ny, self.nx)
        
        # read raw file and store the data
        for i, file_path in enumerate(self.raw_files[idx]):
            frame_data, _, _ = loader(file_path)
            if torch.any(torch.isnan(frame_data)):
                msg = (
                    f'WARNING: NaN values detected in file {file_path}'
                )
                raise ValueError(msg)
                
            data[:,i] = frame_data
                
        # applying in-place data augmentation operations
        if self.data_augmentation is not None:
            self.data_augmentation(data)
        
        pt_file_path = self.processed_files[idx] 
        pt_folder = glob.os.path.dirname(pt_file_path)        
        # creating destination folder if it does not exist
        if (not glob.os.path.exists(pt_folder)):
            glob.os.makedirs(pt_folder)
        
        # save file and update counter (number of saved files)
        torch.save(data, pt_file_path)
        counter += 1
        if counter % (self.__len__() // 10 ) == 0:
            print(f'  {counter:07d}/{self.__len__():07d}')


    def __len__(self):            
        self.__check_setup()
        return self.number_scenes * self.groups_per_scene


    def __getitem__(self, idx):
        """ Method to load one point from the dataset
        
        ----------
        ARGUMENTS
        
            idx: index of the sample (scene index x group index)
            
        ----------
        RETURNS

            _input: tensor with the input frames, format
                    (N, C, D, H, W), where D is the number of
                    frames
            _target: tensor with the target frames, format
                     (N, C, D, H, W), where D is the number
                     frames
        
        """
        pt_file_path = self.processed_files[idx] 
        
        # check if file exists
        self.__check_file__(pt_file_path)
        
        # load file and split between input and target frames
        _data_temp = torch.load(pt_file_path)
        _input = _data_temp[:,:-self.num_target_frames,:,:]
        _target = _data_temp[:,-self.num_target_frames:,:,:]
        
        return _input, _target
    

    def __check_file__(self, file_path):
        """ Check is file exists."""
        if not glob.os.path.isfile(file_path):
            raise ValueError(f'Data file {file_path} does not exist.')
    

    def __check_folder__(self, folder_path):
        """ Check if folder exists."""
        if not glob.os.path.isdir(folder_path):
            raise ValueError(f'Directory {folder_path} does not exist.')            


    def __check_setup(self):
        if not self.setup._called and \
               torch.multiprocessing.current_process().name == 'MainProcess':
            raise ValueError(
                'Method `setup` of pulsenetlib.dataset.PulseNetDataset '
                'must be called before starting the training'
            )


def list_available_files(file_path):
    """ Returns the list of files respecting a format. 
    
    For a file format in a folder, returns the list of files respecting
    the input format (in python format) and an array indicating the
    references of each files, from left to right.
    
    ----------
    ARGUMENTS
    
        file_path: path to files, including format using python
                   string format notation
    
    ----------
    RETURNS
    
        parent_folder: path with the parent folder that contains all data
        list_files: numpy array of sorted files path, arranged by their
                    references. E.g., list_files[2,0] is the path of the file
                    corresponding to the third element of the first reference
                    and the first of the second reference
        array_references: numpy array of integers indicating the
                          index of each reference on the file format
    
    """
    
    # selecting the number of digits on the file format
    indicators_digits = re.findall(r'\D[{:](\d{1,4})[d}]\D', file_path)
    number_references = len(indicators_digits)
    
    # getting the number of digits for each indicator
    number_digits = [int(n_digit) for n_digit in indicators_digits]
    
    # construct a regex from the file format
    raw_file_regex = re.sub(r'\D[{:]\d{1,4}[d}]\D', r'*[0-9]',
                            file_path)
        
    list_files = sorted(glob.glob(raw_file_regex))
    number_files = len(list_files)
    
    # getting the parent folder
    path_before_references = raw_file_regex[:raw_file_regex.find('*[0-9]')]
    parent_folder = glob.os.path.dirname(path_before_references)
    
    min_digits, max_digits = np.min(number_digits), np.max(number_digits)
    
    array_references = np.zeros((number_files, number_references),
                                dtype=np.uint16)
    
    # catching the reference (integer) for each field of the 
    # format, for each file
    digits_regex = '{:d},{:d}'.format(min_digits, max_digits)
    for index_file, file in enumerate(list_files):
        references = re.findall(r'\D(\d{' + digits_regex + r'})\D', file)
        for index_ref, ref in enumerate(references[-number_references:]):
            array_references[index_file,index_ref] = int(ref)
            
    # number of cases per reference
    count_references = np.apply_along_axis(
        lambda x: len(np.unique(x)), axis=0, arr=array_references)
    
    # reshaping the files list such as to reproduce the references
    # organization
    list_files = np.array(list_files, dtype=object)
    list_files = np.reshape(list_files, count_references)
        
    return parent_folder, list_files, array_references
