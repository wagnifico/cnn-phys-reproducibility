#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Handle checkpoint saving.

"""

# native modules
##none##

# third-party modules
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn

# local modules
##none##

class HandleCheckpointSaving(Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, trainer, pl_module):
        # save yaml with k best checkpoints at validation end after last checkpoint
        # (use on_epoch_end instead of on_validation_epoch_end) 
        checkpoint_callbacks = [c for c in
            trainer.callbacks if isinstance(c, ModelCheckpoint)]
    
        epoch = pl_module.current_epoch

        for c in checkpoint_callbacks:
            if epoch % c.period == 0:
                print('Saving yaml with best checkpoints')
                c.to_yaml()

    def on_keyboard_interrupt(self, trainer, pl_module):
        # disable logger and checkpoint when interrupting in the middle on an epoch
        # so that it does not mess logs.
        trainer.logger = None
        rank_zero_warn(f'Detected KeyboardInterrupt: disabling metrics logging '
                       f'for current epoch')
        
        checkpoint_callbacks = [c for c in
            trainer.callbacks if isinstance(c, ModelCheckpoint)]
        
        for c in checkpoint_callbacks:
            c.to_yaml()
            c.period = -1
        
        rank_zero_warn(f'Detected KeyboardInterrupt: disabling model checkpoint '
                       f'for current epoch')

