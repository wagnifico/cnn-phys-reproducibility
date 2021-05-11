#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Module



TODO: Add a short description of the module


"""

# native modules
import os
import inspect
from typing import Callable, Mapping, Optional, Sequence, Tuple

# third-party modules
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import get_init_args

# local modules
import pulsenetlib.transforms
from pytorch_lightning.trainer.states import TrainerState


class PulseNetPL(pl.LightningModule):
    
    def __init__(self, model, optimizer, scheduler, criterion, transforms):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.transforms = transforms
        
        self.hparams.optimizer = optimizer.__class__.__name__
        for i, group in enumerate(self.optimizer.param_groups):
            self.hparams['{0}'.format(i)] = {}
            for key in sorted(group.keys()):
                if key != 'params':
                    self.hparams['{0}'.format(i)][key] = group[key]
        self.hparams.scheduler = scheduler.__class__.__name__

        self.loss_names = self.criterion.get_loss_names()
        scheduler_args = inspect.signature(
            self.scheduler.__init__).parameters.copy()
        
        for name in scheduler_args.keys():
            if name != 'optimizer' and name != 'min_lr':
                self.hparams[name] = self.scheduler.__getattribute__(name) 


    def on_fit_start(self):
        self.hparams.batch_size = self.train_dataloader().batch_size


    def _step(self, batch, batch_idx):
        """ Operations at the step that are common
        to training and validation

        TODO: use decorators or partials to do it smarter
        """
        data, target = batch

        # converting to double precision
        #data = torch.tensor(data, dtype=torch.double)
        #target = torch.tensor(target, dtype=torch.double)
        
        # density shift (in LBM 1 is background density)
        for transform in self.transforms:
            transform(data), transform(target)

        out = self.model(data)
        self.last_prediction = out # used for printing callback

        detailed_loss = self.criterion(out, target)
        loss = torch.mean(torch.stack(detailed_loss))

        dict_metrics = dict(
            zip(self.loss_names, detailed_loss))
       
        return loss, dict_metrics


    def training_step(self, batch, batch_idx):
        loss, dict_metrics = self._step(batch, batch_idx)
        
        return {'loss': loss, 'metrics': dict_metrics}


    def validation_step(self, batch, batch_idx):
        loss, dict_metrics = self._step(batch, batch_idx)
        
        return {'val_loss': loss, 'metrics': dict_metrics}
  

    def _epoch_end(self, phase, output_results):
        """ Operations at end of epoch that are common
        to training and validation

        TODO: use decorators or partials to do it smarter
        """
        pass


    def training_epoch_end(self, training_step_outputs):       
        self.log(
            'loss',
            torch.stack([output['loss'] 
                for output in training_step_outputs]).mean(),
            on_epoch=True
            )
        for key in training_step_outputs[0]['metrics'].keys():
            self.log(
                f'training/{key}',
                torch.stack([output['metrics'][key] 
                    for output in training_step_outputs]).mean(),
                on_epoch=True
                )


    def validation_epoch_end(self, validation_step_outputs):
        self.log(
            'val_loss',
            torch.stack([output['val_loss']
                for output in validation_step_outputs]).mean(),
            on_epoch=True
            )
        for key in validation_step_outputs[0]['metrics'].keys():
            self.log(
                f'validation/{key}',
                torch.stack([output['metrics'][key]
                    for output in validation_step_outputs]).mean(),
                on_epoch=True
                )


    def teardown(self, stage):
       pass
       # self.logger.log_hyperparams(self.hparams)


    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        return [optimizer], [scheduler] 


class PulseNetDatModule(pl.LightningDataModule):
    def __init__(
        self,
        save_dir: str,
        data_format: str,
        labels: Mapping[str, str],
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        channels: Sequence[str],
        frames: Tuple[int, int, int] ,
        max_datapoints: int,
        data_augmentation: Optional[Callable] = None,
        select_mode: Optional[str] = 'start',
    ):
        super().__init__()
        self.save_dir = save_dir
        self.data_format = data_format
        self.labels = labels

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.channels = channels 
        self.number_channels = len(channels)

        self.input_frames = frames[0] 
        self.output_frames = frames[1] 
        self.frame_step = frames[2]

        self.max_datapoints = max_datapoints 
        self.data_augmentation = data_augmentation
        self.select_mode = select_mode


    def preprocess(self, raw_save_dir, raw_format, num_preproc_threads=1):
        import sys

        vtkloader = pulsenetlib.preprocess.VTKLoader(self.channels)
        path_data_format = os.path.join(self.save_dir,
                                        self.labels['training_label'],
                                        self.data_format)
        path_raw_format = os.path.join(raw_save_dir,
                                       self.labels['training_label'],
                                       raw_format)
        
        pulsenetlib.dataset.PulseNetDataset(
                        path_data_format,
                        (self.input_frames, self.output_frames, self.frame_step),
                        self.channels,
                        self.max_datapoints,
                        self.labels['training_label'],
                        self.data_augmentation,
                        preprocess=True,
                        path_raw_format=path_raw_format,
                        rawloader=vtkloader.load2D,
                        num_preproc_threads=num_preproc_threads,
                        select_mode=self.select_mode)
        
        path_data_format = os.path.join(self.save_dir,
                                        self.labels['validation_label'],
                                        self.data_format)
        path_raw_format = os.path.join(raw_save_dir,
                                      self.labels['validation_label'],
                                      raw_format)
        
        pulsenetlib.dataset.PulseNetDataset(
                        path_data_format,
                        (self.input_frames, self.output_frames, self.frame_step),
                        self.channels,
                        self.max_datapoints,
                        self.labels['validation_label'],
                        self.data_augmentation,
                        preprocess=True,
                        path_raw_format=path_raw_format,
                        rawloader=vtkloader.load2D,
                        num_preproc_threads=num_preproc_threads,
                        select_mode=self.select_mode)
        sys.exit()


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            path_data_format = os.path.join(self.save_dir,
                                            self.labels['training_label'],
                                            self.data_format)
            self.train_database = pulsenetlib.dataset.PulseNetDataset(
                            path_data_format,
                            (self.input_frames, self.output_frames, self.frame_step),
                            self.channels,
                            max_data_points=self.max_datapoints,
                            label=self.labels['training_label'],
                            data_augmentation=self.data_augmentation)
            path_data_format = os.path.join(self.save_dir,
                                            self.labels['validation_label'],
                                            self.data_format)
            self.validation_database = pulsenetlib.dataset.PulseNetDataset(
                            path_data_format,
                            (self.input_frames, self.output_frames, self.frame_step),
                            self.channels,
                            max_data_points=int(self.max_datapoints*0.2),
                            label=self.labels['validation_label'],
                            data_augmentation=self.data_augmentation)
        else:
            raise ValueError('test dataset not implemented')


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.train_database,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=self.shuffle,
                    pin_memory=True,
                    drop_last=False)
    

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.validation_database,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False)
