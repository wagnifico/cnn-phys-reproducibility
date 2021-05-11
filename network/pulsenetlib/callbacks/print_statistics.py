#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Callbacks
"""

# native modules
import glob
import os
import datetime

# third-party modules
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

# local modules
##none##


class PrintStatistics(Callback):
    """ Class defining operations that display the evolution training and validation.

    ----------
    ARGUMENTS

        stats_frequency: refresing frequency to print the progress of
                         training/validation in terms o batches in the epoch.
                         Default (1) indicates that a print is performed
                         at the end of every batch. Only applies if verbose
                         is true
        verbose: boolean to select a verboragic class. If True,
                 losses and time stats (average epoch time, remaining time
                 and estimated time of arrival (ETA) are displayed after every
                 epoch end

    """

    def __init__(self,stats_frequency : int = 1,
                 verbose : bool = False ):
        self.stats_frequency = stats_frequency
        self.verbose = verbose
        
        self.num_train_eval = 0
        self.num_valid_eval = 0                

        self.cumulative_epoch_time = datetime.timedelta()


    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_date = datetime.datetime.now() 
        print('\nepoch: {:5d}'.format(pl_module.current_epoch))

       
    def on_train_epoch_start(self, trainer, pl_module):
        self.num_train_eval = 0
        print('| training...')

    
    def on_validation_epoch_start(self, trainer, pl_module):
        self.num_valid_eval = 0
        print('| validation...')

 
    def on_train_batch_end(self, trainer, pl_module, outputs,
                           batch, batch_idx, dataloader_idx):
        if self.verbose:
            self.num_train_eval += batch[0].shape[0]
            if batch_idx % self.stats_frequency == 0:
                self._print_perc(self.num_train_eval,
                                 len(pl_module.train_dataloader().dataset))

               
    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                batch, batch_idx, dataloader_idx):
        if self.verbose:
            self.num_valid_eval += batch[0].shape[0]
            if batch_idx % self.stats_frequency == 0:
                self._print_perc(self.num_valid_eval,
                                 len(pl_module.val_dataloader().dataset))
              

    def _print_perc(self, num_performed, num_total):
        """ Print evolution of run in terms of datapoints """
        print('|   {:05d}/{:05d}'.format(
              num_performed, num_total)
        )


    def on_epoch_end(self, trainer, pl_module):
        """ Print total losses and training time stats """
        ellapsed_time = datetime.datetime.now() - self.epoch_start_date
        self.cumulative_epoch_time += ellapsed_time
       
        average_epoch_time = \
            self.cumulative_epoch_time/(pl_module.current_epoch + 1)
        remaining_time = \
            ((trainer.max_epochs - 1) - pl_module.current_epoch)*average_epoch_time 
        estimated_time_arrival = datetime.datetime.now() + remaining_time
        
        if self.verbose: 
            print('train/val losses: {:12.6e} {:12.6e}'.format(
                trainer.callback_metrics['loss'],trainer.callback_metrics['val_loss'])
            )
            print(('{:^20} '*3 + '\n' + '{:^20} '*3).format(
                'avg epoch time','remaining time','ETA',
                str(average_epoch_time)[:-4],
                str(remaining_time)[:-5],
                str(estimated_time_arrival)[:-7])
            ) 
 

