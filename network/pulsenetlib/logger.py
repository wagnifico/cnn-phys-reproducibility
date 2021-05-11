#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Custom logger funnctionalities

"""

# native modules
import csv
import io
import os
import functools
import operator
from argparse import Namespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# third-party modules
import torch
import numpy as np

from pytorch_lightning import _logger as log
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.base import LightningLoggerBase, merge_dicts
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

# local modules
##none##

class HandleVersion(object):
    r"""
    Handles the version name for experiment logging and model checkpointing.

    Should be called before instatiating loggers.
    
    Args:
        save_dir: Save directory.
        version: experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        resume: resuming from a checkpoint. Resume version must be specified.
    Return:
        logger_kwargs: dictionary with named arguments for loggers ('save_dir', 'name', 'version')
        version: version name
        log_dir: path to log directory
        ckpt_dir: path to checkpoint directory
         
    """
    def __init__(
        self,
        save_dir: str,
        version: Optional[Union[int, str]] = None,
        resume : Optional[bool] = False
    ):
        self._save_dir = save_dir
        self._name = ''
        self._version = version
        self._resume = resume

    def __new__(cls, *args, **kwargs):
        self = super(HandleVersion, cls).__new__(cls)
        self.__init__(*args, **kwargs)

        if os.path.exists(self.log_dir) and not self.resume:
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints', '')
        os.makedirs(ckpt_dir, exist_ok=True)
        name = ''
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        logger_kwargs = {
            'save_dir': self.save_dir,
            'name': name,
            'version':version}

        return logger_kwargs, version, self.log_dir, ckpt_dir 
    
    @property
    def root_dir(self) -> str:
        """
        Parent directory for all checkpoint subdirectories.
        If the experiment name parameter is ``None`` or the empty string, no experiment subdirectory is used
        and the checkpoint will be saved in "save_dir/version_dir"
        """
        if not self.name:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """
        The log directory for this run. By default, it is named
        ``'version_${self.version}'`` but it can be overridden by passing a string value
        for the constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def resume(self) -> bool:
        if os.path.exists(self.log_dir) and self._resume:
            return True
        elif not os.path.exists(self.log_dir) and self._resume:
            raise MisconfigurationException(
            f'Resume logger failed: {self.log_dir} does not exist.'
        )
        else:
            return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self._save_dir, self.name)

        if not os.path.isdir(root_dir):
            log.warning('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class ExperimentWriter(object):
    r"""
    Experiment writer for DiskLogger.

    Currently supports to log hyperparameters and metrics in YAML and numpy 
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """

    NAME_HPARAMS_FILE = 'hparams.yaml'
    NAME_METRICS_FILE = 'metrics.npz'

    def __init__(self, log_dir: str, resume: bool) -> None:
        self.hparams = {}
        self.metrics = []
        self.metrics_keys = []

        self.log_dir = log_dir
        self.resume = resume
        os.makedirs(self.log_dir, exist_ok=True)

        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

        self.results = {}
        
        if self.resume:
            # Check that file exists but don't load it.
            if not os.path.isfile(self.metrics_file_path):
                raise FileNotFoundError(
                f'At resume: no file {self.NAME_METRICS_FILE} found in {self.log_dir}.'
            )

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """Record hparams"""
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics"""
        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        merge_list_dicts_to_dict(
            [metrics],
            self.results
        )

    def log_images(self, tag : str, image, step:Optional[int] = None) -> None:
        if step is None:
            step = 0 

        image_file_path = os.path.join(self.log_dir, 'images')
        os.makedirs(image_file_path, exist_ok=True)
        image.savefig(os.path.join(image_file_path, f'{tag}'))

    def save(self) -> None:
        """Save recorded hparams and metrics into files"""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.results:
            return
        
        if self.resume:
            # load file in npz format and convert it to dict
            loaded_results = dict(np.load(self.metrics_file_path))

            print(self.results) 
            num_correct_elements = np.inf
            for k, v in loaded_results.items():
                # convert np.ndarray type into list
                loaded_results[k] = v.tolist()
                if len(loaded_results[k]) < num_correct_elements:
                    num_correct_elements = len(loaded_results[k])

            for k, v in loaded_results.items():
                # purge orphaned data (e.g. a ctrl+c during training)
                # TODO: check tensorboard
                if len(loaded_results[k]) >= num_correct_elements:
                    del loaded_results[k][num_correct_elements:]
                if k in self.results:
                    # append the new results
                    loaded_results[k].extend(self.results[k])
            
            self.results = loaded_results
            # run in normal mode
            self.resume = False

        #print(self.results)
        np.savez(self.metrics_file_path, **self.results)
    
    def _sanitize_loaded_dict(self, loaded_dict):
        pass


class DiskLogger(LightningLoggerBase):
    r"""
    Log to local file system in yaml and CSV format. Logs are saved to
    ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        version: experiment version. if version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        resume: resuming from a previous checkpoint.
    """

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        resume : Optional[bool] = False
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ''
        self._version = version
        self._resume = resume
        self._experiment = None

    @property
    def root_dir(self) -> str:
        """
        Parent directory for all checkpoint subdirectories.
        If the experiment name parameter is ``None`` or the empty string, no experiment subdirectory is used
        and the checkpoint will be saved in "save_dir/version_dir"
        """
        if not self.name:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """
        The log directory for this run. By default, it is named
        ``'version_${self.version}'`` but it can be overridden by passing a string value
        for the constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir, resume=self.resume)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self.experiment.log_metrics(metrics, step)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def resume(self) -> bool:
        if os.path.exists(self.log_dir) and self._resume:
            return True
        elif not os.path.exists(self.log_dir) and self._resume:
            raise MisconfigurationException(
            f'Resume logger failed: {self.log_dir} does not exist.'
        )
        else:
            return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self._save_dir, self.name)

        if not os.path.isdir(root_dir):
            log.warning('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1



def _handle_value(value):
    print(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value



def merge_list_dicts_to_dict(
        list_dicts: Sequence[Mapping],
        result_dict : Mapping[str, Sequence[float]]
) -> None:
    """
    Concatenates a sequence of dictionaries to a dictionary of lists by using 
    all the keys from the list of dicts and appending values to a list 
    when keys are repeated.

    Args:
        list_dicts: 
            Sequnce of dictionaries to be merged.
        result_dict:
            Dictionary of values
    
    Returns:
        None
    """
    for dict in list_dicts:
        dict = LightningLoggerBase._flatten_dict(dict)
        for list in dict:
            if list in result_dict:
                result_dict[list].append(dict[list])
            else:
                result_dict[list] = [dict[list]]
