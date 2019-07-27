# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: ensemble.py

@time: 2019/4/13 20:39

@desc: ensemble during training process

"""
import os
import math
import keras.backend as K
from keras.callbacks import Callback


class SWA(Callback):
    """
    This callback implements a stochastic weight averaging (SWA) method with constant lr as presented in the paper -
        "Izmailov et al. Averaging Weights Leads to Wider Optima and Better Generalization"
        (https://arxiv.org/abs/1803.05407)
    Author's implementation: https://github.com/timgaripov/swa
    """
    def __init__(self, swa_model, checkpoint_dir, model_name, swa_start=1):
        """
        :param swa_model: the model that we will use to store the average of the weights once SWA begins
        :param checkpoint_dir: the directory where the model will be saved in
        :param model_name: the name of model we're training
        :param swa_start: the epoch when averaging begins. We generally pre-train the network for a certain amount of
                          epochs to start (swa_start > 1), as opposed to starting to track the average from the
                          very beginning.
        """
        super(SWA, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.swa_start = swa_start
        self.swa_model = swa_model   # the model that we will use to store the average of the weights once SWA begins

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.swa_n = 0
        '''Note: I found deep copy of a model with customized layer would give errors'''
        # self.swa_model = copy.deepcopy(self.model)  # make a copy of the model we're training

        '''Note: Something wired still happen even though i use keras.models.clone_model method, so I build swa_model
                 outside this callback and pass it as an argument. It's not fancy, but the best I can do :)
        '''
        # self.swa_model = keras.models.clone_model(self.model)
        self.swa_model.set_weights(self.model.get_weights())    # see: https://github.com/keras-team/keras/issues/1765

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1

        self.epoch += 1

    def update_average_model(self):
        # update running average of parameters
        alpha = 1. / (self.swa_n + 1)
        for layer, swa_layer in zip(self.model.layers, self.swa_model.layers):
            weights = []
            for w1, w2 in zip(swa_layer.get_weights(), layer.get_weights()):
                weights.append((1 - alpha) * w1 + alpha * w2)
            swa_layer.set_weights(weights)

    def on_train_end(self, logs=None):
        print('Logging Info - Saving SWA model checkpoint: %s_swa.hdf5\n' % self.model_name)
        self.swa_model.save_weights(os.path.join(self.checkpoint_dir, '{}_swa.hdf5'.format(self.model_name)))
        print('Logging Info - SWA model Saved')


class SWAWithCLR(Callback):
    """
    SWA with cyclical learning rate, collect model
    """

    def __init__(self, swa_model, checkpoint_dir, model_name, min_lr, max_lr, cycle_length, swa_start=1):
        super(SWAWithCLR, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.swa_start = swa_start
        self.swa_model = swa_model  # the model that we will use to store the average of the weights once SWA begins

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.trn_iteration = 0.
        self.cycle = 0.
        self.swa_n = 0
        self.history = {}

    def on_train_begin(self, logs={}):
        K.set_value(self.model.optimizer.lr, self.max_lr)
        # self.swa_model = keras.models.clone_model(self.model)
        self.swa_model.set_weights(self.model.get_weights())  # see: https://github.com/keras-team/keras/issues/1765

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iteration)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.swa()

    def swa(self):
        t = (self.trn_iteration % self.cycle_length) / self.cycle_length
        lr = (1 - t) * self.max_lr + t * self.min_lr
        K.set_value(self.model.optimizer.lr, lr)

        if t == 0:
            self.cycle += 1
            if self.cycle >= self.swa_start:
                self.update_average_model()
                self.swa_n += 1

    def update_average_model(self):
        # update running average of parameters
        alpha = 1. / (self.swa_n + 1)
        for layer, swa_layer in zip(self.model.layers, self.swa_model.layers):
            weights = []
            for w1, w2 in zip(swa_layer.get_weights(), layer.get_weights()):
                weights.append((1 - alpha) * w1 + alpha * w2)
            swa_layer.set_weights(weights)

    def on_train_end(self, logs=None):
        print('Logging Info - Saving SWA model checkpoint: %s_swa_with_clr.hdf5\n' % self.model_name)
        self.swa_model.save_weights(os.path.join(self.checkpoint_dir, '{}_swa_clr.hdf5'.format(self.model_name)))
        print('Logging Info - SWA model loaded')


class SnapshotEnsemble(Callback):
    """
    This Callback implements `Snapshot Ensemble` method, which can produce an ensemble
    of accurate and diverse models from a single training process.

    SnapShot Ensemble using a cosine annealing learning rate schedule where learning rate starts high and is drooped
    relatively  rapidly to a minimum value near zero before bedding increased again to the maximum. This lr schedule
    is similar as SGDR(https://arxiv.org/pdf/1608.03983.pdf)

    Snapshot Ensemble is proposed by Huang et al. "Snapshot Ensebles: Train 1, Get M for free"
      (https://arxiv.org/pdf/1704.00109.pdf)
    """
    def __init__(self, checkpoint_dir, model_name, max_lr, cycle_length, snapshot_start=1):
        """
        :param checkpoint_dir: where to save the snapshot model
        :param model_name: the name of model we're training
        :param max_lr: upper bound learning rate
        :param cycle_length: the number of iterations (1 mini-batch is 1 iteration) in an cycle, generally we set it to
                             np.ceil(train_iterations / n_cycle)
        :param snapshot_start: epoch to start snapshot ensemble
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.snapshot_start = snapshot_start
        self.n_cycle = 0
        self.trn_iteration = 0
        self.history = {}
        super(SnapshotEnsemble, self).__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.trn_iteration += 1
        self.history.setdefault('iteration', []).append(self.trn_iteration)
        self.history.setdefault('lr', []).append(self.model.optimizer.lr)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.snapshot()

    def snapshot(self):
        # update lr
        fraction_to_restart = self.trn_iteration % self.cycle_length / self.cycle_length
        lr = 0.5 * self.max_lr * (math.cos(fraction_to_restart * math.pi) + 1)
        K.set_value(self.model.optimizer.lr, lr)

        '''Check for the end of cycle'''
        if fraction_to_restart == 0:
            self.n_cycle += 1
            if self.n_cycle >= self.snapshot_start:
                snapshot_id = self.n_cycle - self.snapshot_start
                # print('Logging Info - Iteration %s : Saving Snapshot Ensemble model checkpoint: %s_snapshot_%d.hdf5\n'
                #       % (self.trn_iteration, self.model_name, snapshot_id))
                self.model.save_weights(os.path.join(self.checkpoint_dir, '{}_sse_{}.hdf5'.format(self.model_name,
                                                                                                  snapshot_id)))
                # print('Logging Info - Snapshot Ensemble model Saved')


class HorizontalEnsemble(Callback):
    """
        This Callback implements `Horizontal Ensemble` method, which is proposed by
        Xie et al. "Horizontal and Vertical Ensemble with Deep Representation for Classification"
        (https://arxiv.org/pdf/1306.2759.pdf)
    """
    def __init__(self, checkpoint_dir, model_name, horizontal_start=1):
        """
        :param checkpoint_dir: where to save the snapshot model
        :param model_name: the name of model we're training
        :param horizontal_start: epoch to start horizontal ensemble
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.horizontal_start = horizontal_start
        super(HorizontalEnsemble, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= self.horizontal_start:
            model_id = epoch + 1 - self.horizontal_start
            self.model.save_weights(os.path.join(self.checkpoint_dir,
                                                 '{}_horizontal_{}.hdf5'.format(self.model_name, model_id)))


class FGE(Callback):
    """
    This Callback implement `Fast Geometruc Ensembling` (FGE)
    """
    def __init__(self, checkpoint_dir, model_name, min_lr, max_lr, cycle_length, fge_start=1):
        """
        :param checkpoint_dir: where to save the snapshot model
        :param model_name: the name of model we're training
        :param max_lr: upper bound learning rate
        :param min_lr: lower bound lr
        :param cycle_length: the number of iterations (1 mini-batch is 1 iteration) in an cycle, generally we set it to
                             np.ceil(train_iterations / n_cycle)
        :param fge_start: epoch to start fge ensemble
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.fge_start = fge_start
        self.n_cycle = 0
        self.trn_iteration = 0
        self.history = {}
        super(FGE, self).__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        self.trn_iteration += 1
        self.history.setdefault('iteration', []).append(self.trn_iteration)
        self.history.setdefault('lr', []).append(self.model.optimizer.lr)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.fge()

    def fge(self):
        # update lr
        t = self.trn_iteration % self.cycle_length / self.cycle_length
        if t <= 0.5:
            lr = (1 - 2 * t) * self.max_lr + 2 * t * self.min_lr
        else:
            lr = (2 - 2 * t) * self.max_lr + (2 * t - 1) * self.min_lr
        K.set_value(self.model.optimizer.lr, lr)

        # when the learning rate reaches its minimum value, collect model
        if t == 0.5:
            self.n_cycle += 1
            if self.n_cycle >= self.fge_start:
                fge_id = self.n_cycle - self.fge_start
                self.model.save_weights(os.path.join(self.checkpoint_dir, '{}_fge_{}.hdf5'.format(self.model_name,
                                                                                                  fge_id)))


class PolyakAverage(Callback):
    """
    Same as swa (https://arxiv.org/abs/1803.05407), this callback combine the weights from multiple different models
    into a single model for making predictions by using an average of the weights from models seen toward the end of
    the training run, which can be further improved by  using linearly or exponentially decreasing weighted average.

    Reference: https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/
    """
    def __init__(self, polyak_model, temp_polyak_model, checkpoint_dir, model_name, avg_type, early_stop_patience,
                 polyak_start=1):
        """
        :param polyak_model: the model that we will use to store the average of the weights
        :param checkpoint_dir: the directory where the model will be saved in
        :param model_name: the name of model we're training
        :param polyak_start: the epoch when averaging begins. We generally pre-train the network for a certain amount of
                             epochs to start (polyak_start > 1), as opposed to starting to track the average from the
                             very beginning.
        """
        super(PolyakAverage, self).__init__()
        self.polyak_model = polyak_model
        self.temp_polyak_model = temp_polyak_model
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.avg_type = avg_type
        assert avg_type in ['avg', 'linear', 'exp']
        self.early_stop_patience = early_stop_patience
        self.polyak_start = polyak_start

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.polyak_n = 0

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch + 1) >= self.polyak_start:
            self.model.save_weights(os.path.join(self.checkpoint_dir, '{}_polyak_{}_{}.hdf5'.format(self.model_name, self.avg_type, self.polyak_n)))
            self.polyak_n += 1

        self.epoch += 1

    def on_train_end(self, logs=None):
        self.polyak_n -= self.early_stop_patience
        if self.avg_type == 'avg':
            contrib = [1.0 / self.polyak_n for _ in range(self.polyak_n)]
        elif self.avg_type == 'linear':
            contrib = [i / self.polyak_n for i in range(1, self.polyak_n+1)]
            norm = sum(contrib)
            contrib = [i / norm for i in contrib]
        elif self.avg_type == 'exp':
            alpha = 2.0
            contrib = [math.exp(-i/alpha) for i in range(self.polyak_n, 0, -1)]
            norm = sum(contrib)
            contrib = [i / norm for i in contrib]
        else:
            raise ValueError('avg_type not understood: {}'.format(self.avg_type))
        for polyak_id in range(self.polyak_n):
            print('Logging Info - Loading Polyak model checkpoint: {}_polyak_{}_{}.hdf5'.format(self.model_name, self.avg_type, polyak_id))
            self.temp_polyak_model.load_weights(os.path.join(self.checkpoint_dir, '{}_polyak_{}_{}.hdf5'.format(self.model_name, self.avg_type, polyak_id)))
            for layer, temp_layer in zip(self.polyak_model.layers, self.temp_polyak_model.layers):
                weights = []
                for w1, w2 in zip(layer.get_weights(), temp_layer.get_weights()):
                    if polyak_id == 0:
                        weights.append(contrib[polyak_id] * w2)
                    else:
                        weights.append(w1 + contrib[polyak_id] * w2)
                layer.set_weights(weights)
            os.remove(os.path.join(self.checkpoint_dir, '{}_polyak_{}_{}.hdf5'.format(self.model_name, self.avg_type, polyak_id)))

        for polyak_id in range(self.polyak_n, self.polyak_n+self.early_stop_patience):
            os.remove(os.path.join(self.checkpoint_dir, '{}_polyak_{}_{}.hdf5'.format(self.model_name, self.avg_type, polyak_id)))

        print('Logging Info - Saving Polyak model checkpoint: %s_polayk.hdf5\n' % self.model_name)
        self.polyak_model.save_weights(os.path.join(self.checkpoint_dir, '{}_polyak_{}.hdf5'.format(self.model_name, self.avg_type)))
        print('Logging Info - Polyak model Saved')

