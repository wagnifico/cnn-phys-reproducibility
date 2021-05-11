#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, importlib, warnings

# neural network base functions and classes
import pulsenetlib.dataset
import pulsenetlib.optimizer
import pulsenetlib.scheduler
import pulsenetlib.model
import pulsenetlib.neuralnet2d
import pulsenetlib.criterion

if importlib.util.find_spec('mpi4py') is not None:
    import pulsenetlib.tester
else:
    if is_main:
        warnings.formatwarning =\
             lambda message,category,filename,lineno,file : str(message)
        warnings.warn(
            'Tester module is based on mpi4py, unavailable in the '+
            'current environment. You must install mpi4py to use it.\n\n',
            category=UserWarning, stacklevel=0
        )


# for dealing with the fields
import pulsenetlib.transforms
import pulsenetlib.tools


# training parameters and dealing with data
import pulsenetlib.arguments
import pulsenetlib.preprocess


# for loading/exporting models and plotting results
import pulsenetlib.datatransfer
import pulsenetlib.visualize


# lightning and callback hooks in pytorch-lightning
import pulsenetlib.logger
import pulsenetlib.lightning
import pulsenetlib.callbacks


# physics of the problem
import pulsenetlib.acoustics


# function utilities
import pulsenetlib.utilities


