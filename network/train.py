#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""

Sample code for the training. 

To run, a configuration file is demanded:
python train.py --trainingConf configuration.xml


if calling from a different folder, include
'sys.path.append(path/to/pulsenetlib)' before importing the
module.

Add --preproc option to do preprocessing.

"""

# native modules
import glob
import os
import sys
import argparse
import datetime
import copy
import functools
import importlib.util

# third party modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pytorch_lightning as pl

# local modules
import pulsenetlib


def cli_main():

    pl.seed_everything(12)
    
    #****************** Arguments and inputs *************************
    # parsing execution arguments
    parser = argparse.ArgumentParser(
                description='Training script.',
                formatter_class=pulsenetlib.arguments.SmartFormatter
    )
    arguments = pulsenetlib.arguments.parse_arguments(parser)

    # reading configuration file. script arguments overwrite the
    # values on configuration file
    training_configuration, resume, verbose, is_double = \
        pulsenetlib.arguments.read_arguments(arguments)
    
    preprocess = arguments.preproc

    if is_double:
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

    if torch.cuda.is_available():
        print('\nActive CUDA Device: GPU ',
              torch.cuda.current_device())    
        run_device = torch.device('cuda:0')
        number_gpus = 1
    else:
        print('\nNot using GPU.')
        run_device = torch.device('cpu')
        number_gpus = 0

    # print training header
    print('\n\n' + 72*'-' + '\n')
    print('Launching training')
    print(datetime.datetime.now(),'\n')
    
    
    
    # creating destination folder, if is not there
    model_path = os.path.realpath(training_configuration['modelDir'])
    if not os.path.isdir(training_configuration['modelDir']):
        print(f'Creating destination folder:\n'
              f'{model_path}\n')
        os.makedirs(model_path, exist_ok=False)
        
    # split the model configuration as an unique dictionary and 
    # remove it from training_configuration
    model_configuration = copy.deepcopy(
            training_configuration['modelParam'])
    del training_configuration['modelParam']
    
    version = training_configuration['version']
    logger_kwargs, version, log_dir, ckpt_dir = \
         pulsenetlib.logger.HandleVersion(model_path,version,resume)
    training_configuration['version'] = version
    
    if resume:
        print('\n{:^72}\n'.format('#RESTARTING#TRAINING#').\
              replace(' ','-').replace('#',' '))
        
        # load configuration files from checkpoint
        temp_training_configuration, temp_model_configuration = \
            pulsenetlib.datatransfer.resume_configuration_files(log_dir)
        
        # overwriting model and its configuration from checkpoint
        model_configuration.update(temp_model_configuration)
        loaded_module_spec = \
            pulsenetlib.datatransfer.load_model_spec(log_dir)
        
        # overwritting the parameters passed as arguments, if any
        training_configuration, resume, verbose = \
            pulsenetlib.arguments.read_arguments(
                arguments, True, training_configuration)
    else:
        # Create a copy the model.py into the model's folder,
        # so that we don't lose the network architecture. Loaded
        # when resuming a training.
        pulsenetlib.datatransfer.save_model(log_dir)        

    # save configuration files to log folder
    pulsenetlib.datatransfer.save_configuration_files(
        log_dir, training_configuration, model_configuration)
    
    #************************* DataModule *****************************
    channels = model_configuration['channels']
    number_channels = len(channels)
    input_frames = model_configuration['numInputFrames']
    output_frames = model_configuration['numOutputFrames']
    frame_step = training_configuration['frameStep']
    select_mode = training_configuration['selectMode']
    
    # selecting the data augmentation operations, performed just
    # after reading the data files
    if training_configuration['flipData']:
        data_augmentation = pulsenetlib.transforms.RandomFlip()
    else:
        data_augmentation = None

    labels = {'training_label': 'training',
              'validation_label': 'validation'}
        
    data_module = pulsenetlib.lightning.PulseNetDatModule(
                        save_dir=training_configuration['dataPath'],
                        data_format=training_configuration['formatPT'],
                        labels=labels,
                        batch_size=training_configuration['batchSize'],
                        num_workers=training_configuration['numWorkers'],
                        shuffle=training_configuration['shuffleTraining'],
                        channels=channels,
                        frames=(input_frames, output_frames, frame_step),
                        max_datapoints=training_configuration['maxDatapoints'],
                        data_augmentation=data_augmentation,
                        select_mode=select_mode,
                    ) 
    if preprocess:
        data_module.preprocess(
                        raw_save_dir=training_configuration['dataPathRaw'],
                        raw_format=training_configuration['formatVTK'],
                        num_preproc_threads=1
                    )
    data_module.setup('fit')
        
    #************************ Data transforms *******************
    scalar_add = torch.tensor(
        model_configuration['scalarAdd']).to(device=run_device)

    transforms= []
    transforms.append(
        functools.partial(pulsenetlib.transforms.offset_,
                          scalars=scalar_add)
    )
                                  
    #************************* Create the model **********************
    print('\n{:^72}\n'.format('#MODEL#')\
          .replace(' ','-').replace('#',' '))
    
    # TODO: coupled our resume functionalities with PL   
    std_norm = torch.tensor(
        model_configuration['stdNorm']).to(device=run_device)
    avg_remove = torch.tensor(
        model_configuration['avgRemove']).to(device=run_device)
        
    # create or load model
    if not resume:
        # define neural network
        net = pulsenetlib.model.PulseNet(number_channels,
                                         input_frames,
                                         output_frames,
                                         std_norm,
                                         avg_remove)        
    else:
        print('Load module used for generating the loaded state')
        print('  {:}'.format(loaded_module_spec.origin))
        
        loaded_module = importlib.util.module_from_spec(
            loaded_module_spec)
        # load module
        loaded_module_spec.loader.exec_module(loaded_module)
        
        # define neural network from loaded module
        net = loaded_module.PulseNet(number_channels,
                                     input_frames,
                                     output_frames,
                                     std_norm,
                                     avg_remove)


    # printing model
    print('Neural network:\n')
    print(net)

    # This can be put inside a callback    
    # initialize network weights with Kaiming normal method (a.k.a MSRA)
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_uniform_(m.weight)
            #torch.nn.init.xavier_normal_(m.weight)
            #torch.nn.init.normal_(m.weight, mean=-1e-4, std=0.06)
            #torch.nn.init.constant_(m.weight, .01)

    if not resume:
        net.apply(init_weights)

    # This can be put inside a callback    
    print('\nLoaded/initiated weights')
    list_parameters = torch.cat(
        [p.view(-1) for p in net.parameters() if p.requires_grad])
    
    print('Trainable parameters statistics:\n')
    print('  number: {:}'.format(list_parameters.numel()))
    print('  AVG: {:+.3e}'.format(list_parameters.mean()))
    print('  STD: {:+.3e}'.format(list_parameters.std()))
        
    # clean variable
    del list_parameters
        
    #********************** Optimizer definition ***********************
    print('\n{:^72}\n'.format('#OPTIMIZER#').\
          replace(' ','-').replace('#',' '))
    
    if training_configuration['optimizer']['lr'] == 'auto':
        lr = 1e-3
        auto_lr_finder = True
    else:
        lr = training_configuration['optimizer']['lr']
        auto_lr_finder = False
    wd = training_configuration['optimizer']['weight_decay']
    
    #optimizer = torch.optim.Adam(
    #    net.parameters(), lr=lr, weight_decay=wd)
    optimizer = pulsenetlib.optimizer.RAdam(
        net.parameters(), lr=lr, weight_decay=wd)

    for param_group in optimizer.param_groups:
        print('Initial Learning Rate of optimizer = {:}'.format(
            str(param_group['lr'])))

    #********************** Scheduler definition ***********************
    print('\n{:^72}\n'.format('#SCHEDULER#').\
          replace(' ','-').replace('#',' '))
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **training_configuration['scheduler']['reduceLROnPlateau'])

    # print scheduler parameters
    print('Scheduler : Reduce Learning Rate on Plateau')
    print('Scheduler params:\n')
    for key, value in \
        training_configuration['scheduler']['reduceLROnPlateau'].items():
        print('   {:} = {:}'.format(key,value))
    print()
    
    #************************ Define Criterion *******************
    print('\n{:^72}\n'.format('#DEFINING#TRAINING#CRITERIA#')\
          .replace(' ','-').replace('#',' '))
    
    criterion = pulsenetlib.criterion.Criterion(
        model_configuration['L2Lambda'],
        model_configuration['GDL2Lambda'],
        run_device)
    
    criterion.summary()
    
    #********************** Data Statistics **********************
    print('\n{:^72}\n'.format('#DATA#STATISTICS#')\
          .replace(' ','-').replace('#',' '))
    
    for name, loader in zip(
                        ('Training','Validation'),
                        (
                            data_module.train_dataloader(),
                            data_module.val_dataloader()
                        )
                    ):
        print(f'{name}:\n'
              f'  batch size: {loader.batch_size:d}\n'
              f'  number of batches: {len(loader):d}\n')
       
    # calculating mean and std of the training data, complete
    # database
    data_mean, data_std = \
        pulsenetlib.tools.online_mean_and_sd(data_module.train_dataloader())
    
    print('Calculating first and second moments...\n')
    for index_channel, channel in enumerate(channels):
        print('  {:}:'.format(channel))
        print('    AVG [data, gradx, grady]: ' +
              '[{:+.3e}, {:+.3e}, {:+.3e}]'.format(
                *data_mean[index_channel,:]))
        print('    STD [data, gradx, grady]: ' +
              '[{:+.3e}, {:+.3e}, {:+.3e}]'.format(
                *data_std[index_channel,:]))
        print('\n')
    
    callbacks = []
    
    stats_frequency = min(
        [len(data_module.train_dataloader()),
         len(data_module.val_dataloader())]) // 3 
    if stats_frequency == 0: stats_frequency += 1
 
    callbacks.append(
        pulsenetlib.callbacks.PrintStatistics(
            stats_frequency, verbose=verbose)
    )
    callbacks.append(
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    )

    callbacks.append(
        pulsenetlib.callbacks.HandleCheckpointSaving()
    )
    
    trainer_kwargs = {'max_epochs': training_configuration['maxEpochs'],
                      'callbacks': callbacks,
                      'gpus': number_gpus,
                      'progress_bar_refresh_rate': 0,
                      'default_root_dir': log_dir}                 

    # when using a list of loggers, lightning automatically concatenates all
    # `name` and `version`, for usage in model_checkpoint callback.
    # To prevent this, we force all the loggers to have the same version and
    # name in the model_checkpoint callback, and we pass a custom checkpoint
    # path to the ModelCheckpoint callback.

    logger_name = training_configuration['logger']

    logger = []
    if 'np' in logger_name:
        print('Using disk numpy logger\n')
        logger.append(
            pulsenetlib.logger.DiskLogger(
                resume=resume, **logger_kwargs)
        )
    if 'tb' in logger_name:
        print('Using tensorboard logger\n')
        logger.append(
            pl.loggers.TensorBoardLogger(
                **logger_kwargs)
    )
    if not logger_name:
        raise ValueError(f'logger list ({logger_name}) cannot be empty') 

    trainer_kwargs.update({'logger': logger})

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=ckpt_dir,
        monitor='val_loss',
        save_top_k=training_configuration['saveTopK'],
        save_last=False,
        period=training_configuration['freqToFile'] \
            if logger_name == 'np' else 1,
        verbose=verbose
    )

    trainer_kwargs.update({'checkpoint_callback': checkpoint_callback })

    if resume:
        # TODO: for now, load the most recent ckpt. We should 
        # handle loading the "best" model.
        path_to_ckpt = pulsenetlib.datatransfer.load_pl_ckpt(ckpt_dir)
        trainer_kwargs.update(
            {'resume_from_checkpoint': path_to_ckpt }
        )
    
    pulsenet = pulsenetlib.lightning.PulseNetPL(
                   model=net,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   criterion=criterion,
                   transforms=transforms)
    
    trainer = pl.Trainer(**trainer_kwargs)

    if auto_lr_finder:
        lr_finder_result_path = os.path.join(log_dir, 'lr_finder.png')
        print(f'Executing lr finding, results written in {lr_finder_result_path} \n' 
              f'warning: this is not necesarily the best LR for training '
              f'the model, but a good starting point for further lr tuning')

        # use dummy data module to spped up lr finder
        # either lower batch_size or change max_datapoints
        dummy_data_module = pulsenetlib.lightning.PulseNetDatModule(
                    save_dir=training_configuration['dataPath'],
                    data_format=training_configuration['formatPT'],
                    labels=labels,
                    batch_size=2,
                    num_workers=training_configuration['numWorkers'],
                    shuffle=training_configuration['shuffleTraining'],
                    channels=channels,
                    frames=(input_frames, output_frames, frame_step),
                    max_datapoints=training_configuration['maxDatapoints'],
                    data_augmentation=data_augmentation,
                    select_mode=select_mode
                )
        dummy_data_module.setup('fit')

        lr_finder = trainer.tuner.lr_find(pulsenet,
            dummy_data_module.train_dataloader(),
            mode='exponential',
            min_lr=1e-6,
            max_lr=1e-3,
            num_training=100)

        fig = lr_finder.plot(suggest=True)
        fig.savefig(lr_finder_result_path)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print(f'Suggested lr = {new_lr}')
        # update hparams of the model
        pulsenet.hparams.lr = new_lr
      
    trainer.fit(pulsenet,
                data_module)


if __name__ == "__main__":
    cli_main()


