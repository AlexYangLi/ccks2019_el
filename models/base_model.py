# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: base_model.py

@time: 2019/5/12 9:19

@desc:

"""

from keras.callbacks import *
from callbacks.ensemble import *
from callbacks.lr_scheduler import *


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.callbacks = []
        self.model = None
        self.swa_model = None

    def add_model_checkpoint(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))
        print('Logging Info - Callback Added: ModelCheckPoint...')

    def add_early_stopping(self):
        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))
        print('Logging Info - Callback Added: EarlyStopping...')

    def add_clr(self, kind, min_lr, max_lr, cycle_length):
        """
        add cyclic learning rate schedule callback
        :param kind: add what kind of clr, 0: the original cyclic lr, 1: the one introduced in FGE, 2: the one
                     introduced in swa
        """
        if kind == 0:
            self.callbacks.append(CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=cycle_length/2, mode='triangular2',
                                           plot=True, save_plot_prefix=self.config.exp_name))
        elif kind == 1:
            self.callbacks.append(CyclicLR_1(min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length, plot=True,
                                             save_plot_prefix=self.config.exp_name))
        elif kind == 2:
            self.callbacks.append(CyclicLR_2(min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length, plot=True,
                                             save_plot_prefix=self.config.exp_name))
        else:
            raise ValueError('param `kind` not understood : {}'.format(kind))
        print('Logging Info - Callback Added: CLR_{}...'.format(kind))

    def add_sgdr(self, min_lr, max_lr, cycle_length):
        self.callbacks.append(SGDR(min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length,
                                   save_plot_prefix=self.config.exp_name))
        print('Logging Info - Callback Added: SGDR...')

    def add_swa(self, with_clr, min_lr=None, max_lr=None, cycle_length=None, swa_start=5):
        if with_clr:
            self.callbacks.append(SWAWithCLR(self.swa_model, self.config.checkpoint_dir, self.config.exp_name,
                                             min_lr=min_lr, max_lr=max_lr, cycle_length=cycle_length,
                                             swa_start=swa_start))
        else:
            self.callbacks.append(SWA(self.swa_model, self.config.checkpoint_dir, self.config.exp_name,
                                      swa_start=swa_start))
        print('Logging Info - Callback Added: SWA with {}...'.format('clr' if with_clr else 'constant lr'))

    def add_sse(self, max_lr, cycle_length, sse_start):
        self.callbacks.append(SnapshotEnsemble(self.config.checkpoint_dir, self.config.exp_name,
                                               max_lr=max_lr, cycle_length=cycle_length, snapshot_start=sse_start))
        print('Logging Info - Callback Added: Snapshot Ensemble...')

    def add_fge(self, min_lr, max_lr, cycle_length, fge_start):
        self.callbacks.append(FGE(self.config.checkpoint_dir, self.config.exp_name, min_lr=min_lr, max_lr=max_lr,
                                  cycle_length=cycle_length, fge_start=fge_start))
        print('Logging Info - Callback Added: Fast Geometric Ensemble...')

    def init_callbacks(self, data_size):
        cycle_length = 4 * math.floor(data_size / self.config.batch_size)

        if 'modelcheckpoint' in self.config.callbacks_to_add:
            self.add_model_checkpoint()
        if 'earlystopping' in self.config.callbacks_to_add:
            self.add_early_stopping()
        if 'clr' in self.config.callbacks_to_add:
            self.add_clr(kind=0, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'sgdr' in self.config.callbacks_to_add:
            self.add_sgdr(min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'clr_1' in self.config.callbacks_to_add:
            self.add_clr(kind=1, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'clr_2' in self.config.callbacks_to_add:
            self.add_clr(kind=2, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length)
        if 'swa' in self.config.callbacks_to_add:
            self.add_swa(with_clr=False, swa_start=self.config.swa_start)
        if 'swa_clr' in self.config.callbacks_to_add:
            self.add_swa(with_clr=True, min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length,
                         swa_start=self.config.swq_clr_start)
        if 'sse' in self.config.callbacks_to_add:
            self.add_sse(max_lr=self.config.max_lr, cycle_length=cycle_length, sse_start=self.config.sse_start)
        if 'fge' in self.config.callbacks_to_add:
            self.add_fge(min_lr=self.config.min_lr, max_lr=self.config.max_lr, cycle_length=cycle_length,
                         fge_start=self.config.fge_start)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def load_model(self, filename):
        # we only save model's weight instead of the whole model
        self.model.load_weights(filename)

    def load_best_model(self):
        print('Logging Info - Loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)))
        print('Logging Info - Model loaded')

    def load_swa_model(self, swa_type='swa'):
        print('Logging Info - Loading SWA model checkpoint: %s_%s.hdf5\n' % (self.config.exp_name, swa_type))
        self.load_model(os.path.join(self.config.checkpoint_dir, '%s_%s.hdf5' % (self.config.exp_name, swa_type)))
        print('Logging Info - SWA Model loaded')

    def summary(self):
        self.model.summary()