#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for dealing with the script arguments and training parameters.

TODO: add a function to simply update the training dictionary with the
arguments
        
"""

# native modules
import argparse

# third party modules
import yaml

# local modules
##none##


class SmartFormatter(argparse.HelpFormatter):
    """ Class with the argument parser format
    """
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

    
    
def parse_arguments(parser):
    """ Argument parser function.
    
    ----------
    ARGUMENTS
    
        parser: argument parser
        
    ----------    
    RETURNS
    
        arguments: parsed arguments
    
    """
    
    parser.add_argument(
        '--trainingConf',
        default='trainConfig.yaml',
        help='R|Training yaml config file.\n'
             '  Default: trainConfig.yaml')
    parser.add_argument(
        '--modelDir',
        help='R|Output folder location for trained model.\n'
             'When resuming, reads from this location.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--modelFilename',
        help='R|Model name.\n'
        '  Default: written in trainingConf file.')
    parser.add_argument(
        '--dataDir',
        help='R|Dataset location.\n'
        '  Default: written in trainingConf file.')
    parser.add_argument(
        '--resume', action="store_true",
        default=False,
        help='R|Resumes training from checkpoint in modelDir.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--version', type=str,
        help='R|Logger version. Creates a folder: "modelDir/version"\n'
            '  Default: None. If version is not specified the logger inspects the modelDir \n'
            '  directory for existing versions, then automatically assigns the next available version.\n'
            '  If resume is set to True, loads from version checkpoing or the last available version.')
    parser.add_argument(
        '--batchSize', type=int,
        help='R|Batch size for training.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--maxEpochs', type=int,
        help='R|Maximum number training epochs.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--maxDatapoints', type=int,
        help='R|Maximum number of datapoints.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--shuffle', action="store_true",
        help='R|Shuffle dataset when training.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--lr', type=str,
        help='R|Learning rate.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--numWorkers', type=int,
        help='R|Number of parallel workers for dataset loading.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--freqToFile', type=int,
        help='R|Epoch frequency for loss output to file/image saving.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--saveTopK', type=int,
        help='R|Save k best validation loss models during training.\n'
             '  Default: written in trainingConf file.')
    parser.add_argument(
        '--verbose', action="store_true",
        default=False,
        help='R|Prints detailed information throughout the run.\n'
             '  Default: False.')
    parser.add_argument(
        '--double', action="store_true",
        default=False,
        help='R|Use double precision (64-bits).\n'
             '  Default: False.')
    parser.add_argument(
        '--preproc', action="store_true",
        default=False,
        help='R|Perform preprocessing or run trainin,g.\n'
             '  Default: False.')
    parser.add_argument(
        '--logger', type=str, action="append",
        help='R|Logger set by user.\n'
             '  "np": saves logs in numpy format. \n'
             '  "tb": saves logs in tensorboard format  \n'
             '  Default: np')


    # check arguments
    print('Parsing and checking arguments\n')
    return parser.parse_args()



def read_arguments(arguments, resume=False, conf_resume=None):
    """ Reading the training configuration file.
        
    When parameters are defined both in configuration
    file and in the command, priority is given to the
    command options.
    
    ----------
    ARGUMENTS
        
        arguments: main script parsed arguments
        resume: set to True when resuming training
        
    ----------
    RETURNS
    
        conf: updated configuration dictionnary
        resume: boolean indicating if one must resume a 
                training
        verbose: boolean indicating if verbose run
                 (print details for each epoch)
        is_double: boolean indicating if run will be made in double precision
                   (64-bits)
    
    """
    
    # reading parameters file
    if not resume:
        with open(arguments.trainingConf, 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
    else:
        conf = conf_resume

    # overwriting parameters defined in file, if
    # provided on command.
    for parameter in ['modelDir', 'modelFilename', 'dataDir', 
                      'batchSize', 'maxEpochs', 'maxDatapoints', 
                      'numWorkers', 'freqToFile', 'saveTopK']:
        argument = getattr(arguments, parameter)
        if argument is not None:
            conf[parameter] = argument
        else:
            pass # keep conf defaults
    
    if arguments.version is not None: conf['version'] = arguments.version
    if arguments.shuffle: conf['shuffleTraining'] = True
    if arguments.lr is not None: conf['optimizer']['lr'] = arguments.lr
    
    try:
        conf['optimizer']['lr'] = float(conf['optimizer']['lr'])
    except ValueError:
        if not conf['optimizer']['lr'] == 'auto':
            raise ValueError(
                f'Argument lr ({conf["optimizer"]["lr"]}) must be either "auto" or float number')
    
    if not arguments.resume:
        resume = conf['resumeTraining']
    else:
        resume = arguments.resume
    
    # force to preproc
    if arguments.preproc:
        print('Running preprocessing only')
        resume = False        
    
    verbose = arguments.verbose
    is_double = arguments.double

    logger_name = []
    if arguments.logger is not None:
        for name in arguments.logger:
            if name in ('np', 'tb'):
                logger_name.append(name)
    else:
        logger_name = conf['logger']
    if not logger_name:
        # if empty, fallback to default 
        logger_name = ['np']
    conf['logger'] = logger_name
    
    return conf, resume, verbose, is_double

