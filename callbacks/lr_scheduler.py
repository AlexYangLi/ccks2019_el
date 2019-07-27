# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: lr_scheduler.py

@time: 2019/4/13 20:28

@desc: different learning rate scheduler

"""
import os
import math
import numpy as np
import keras.backend as K
from keras.callbacks import Callback

from config import IMG_DIR

# Force matplotlib to not use any Xwindows backend.
# See: https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LRScheduler(Callback):
    def __init__(self, alpha=0.2, sma=20, plot=False, save_plot_prefix=None):
        """
        :param alpha: parameter for exponential moving average
        :param sma: number of batches for simple moving average
        :param plot: whether to plot figures to show the results of the experiment
        :param save_plot_prefix: where to save the figure
        """
        super(LRScheduler, self).__init__()
        self.history = {}
        self.alpha = alpha
        self.sma = sma
        self.plot = plot
        self.save_plot_prefix = save_plot_prefix

    def update_lr(self):
        raise NotImplementedError

    def on_train_end(self, logs=None):
        if self.plot:
            self.plot_lr()
            self.plot_loss(valid_loss=False)    # plot train loss
            self.plot_loss(valid_loss=True)     # plot valid loss
            self.plot_acc(valid_acc=False)      # plot train acc
            self.plot_acc(valid_acc=True)       # plot valid acc

    def plot_lr(self):
        """
        plot the learning rate w.r.t iteration
        """
        if 'iteration' in self.history:
            iteration = self.history['iteration']
            lr = self.history['lr']
            if self.save_plot_prefix:
                lr_save_path = os.path.join(IMG_DIR, self.save_plot_prefix+'_lr.png')
            else:
                lr_save_path = None
            self.plot_figure(iteration, lr, 'iteration', 'learning rate', yscale='log', save_path=lr_save_path)

    def plot_loss(self, valid_loss: bool):
        """
        plot the loss w.r.t learning rate
        """
        loss_key = 'val_loss' if valid_loss else 'loss'
        if loss_key not in self.history:
            return
        lr = self.history['lr']
        loss = self.history[loss_key]
        avg_loss = self.exp_move_avg(loss, self.alpha)

        loss_derivates = [0] * self.sma
        for i in range(self.sma, len(loss)):
            loss_derivates.append((loss[i] - loss[i - self.sma]) / self.sma)

        if self.save_plot_prefix:
            loss_save_path = os.path.join(IMG_DIR, self.save_plot_prefix+'_%s_lr.png' % loss_key)
            avg_loss_save_path = os.path.join(IMG_DIR, self.save_plot_prefix+'_avg_%s_lr.png' % loss_key)
            loss_derivate_save_path = os.path.join(IMG_DIR, self.save_plot_prefix+'_%s_derivate_lr.png' % loss_key)
        else:
            loss_save_path, avg_loss_save_path, loss_derivate_save_path = None, None, None

        self.plot_figure(lr, loss, 'learning rate', 'loss', xscale='log', save_path=loss_save_path)
        self.plot_figure(lr, avg_loss, 'learning rate', 'exp_move_avg loss',  xscale='log',
                         save_path=avg_loss_save_path)
        self.plot_figure(lr, loss_derivates, 'learning rate', 'rate of loss change',  xscale='log',
                         save_path=loss_derivate_save_path)

    def plot_acc(self, valid_acc: bool):
        """
        plot the accuracy with respect to learning rate
        """
        acc_key = 'val_acc' if valid_acc else 'acc'
        if acc_key in self.history:
            lr = self.history['lr']
            acc = self.history[acc_key]
            avg_acc = self.exp_move_avg(acc, self.alpha)
            if self.save_plot_prefix:
                acc_save_path = os.path.join(IMG_DIR, self.save_plot_prefix+'_%s_lr.png' % acc_key)
                avg_acc_save_path = os.path.join(IMG_DIR, self.save_plot_prefix+'_avg_%s_lr.png' % acc_key)
            else:
                acc_save_path, avg_acc_save_path = None, None
            self.plot_figure(lr, acc, 'learning rate', 'acc',  xscale='log', save_path=acc_save_path)
            self.plot_figure(lr, avg_acc, 'learing_rate', 'exp_move_avg acc',  xscale='log', save_path=avg_acc_save_path)

    @staticmethod
    def plot_figure(x, y, xlabel, ylabel, xscale=None, yscale=None, show=False, save_path=None):
        plt.clf()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.plot(x, y)

        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)

        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
            print('Logging Info - Plot Figure has save to:', save_path)

    @staticmethod
    def exp_move_avg(_list, alpha):
        """exponential moving average"""
        cur_value = 0
        avg_list = []
        for i in _list:
            cur_value = (1-alpha)*cur_value + alpha*i
            avg_list.append(cur_value)
        avg_list[0] = avg_list[1]
        return avg_list


# Note: Modify based on https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py
class LRRangeTest(LRScheduler):
    def __init__(self, num_batches, start_lr=1e-5, end_lr=1., alpha=0.2, sma=20, plot=True, save_plot_prefix=None):
        """
        This callback implements LR Range Test as presented in section 3.3 of the 2015 paper -
            "Cyclical Learning Rates for Training Neural Networks" (https://arxiv.org/abs/1506.01186), which is used to
        estimate an optimal learning rate for dnn.

        This idea behind is to train a network starting from a low learning rate and increase the learning rate
        exponentially for every mini-batch. Record the learning rate and training loss for every batch. Then, plot the
        loss (also the rate of change of the loss) and the learning rate. Then select a lr range with the fastest
        decrease in the loss.

        For more detail, please see the paper and this blog:
        https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0.

        # Example
            ```python
                lrtest = LRRangeTest(num_batches=128)
                model.fit(X_train, Y_train, callbacks=[lrtest])
            ```
        :param num_batches: num of iterations to run lr range test
        :param start_lr: the lower bound of the lr range
        :param end_lr: the upper bpound of the lr range
        :param alpha: parameter for exponential moving average
        :param sma: number of batches for simple moving average
        :param plot: whether to plot figures to show the results of the experiment
        :param save_plot_prefix: where to save the figure
        """
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))     # exponentially increase
        self.best_loss = 1e9
        self.trn_iterations = 0
        super(LRRangeTest, self).__init__(alpha, sma, plot,
                                          save_plot_prefix + '_lr_range_test' if save_plot_prefix else None)

    def on_train_begin(self, logs=None):
        # set initial learning rate
        K.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iteration', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        loss = logs['loss']
        if math.isnan(loss) or loss > self.best_loss * 4:
            # stop training whe the loss gets a lot higher than the previously observed best value
            print('Loggin Info - LR Range Test training end!')
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss
        self.update_lr()

    def update_lr(self):
        # increse the learning rate exponentially for next batch
        lr = K.get_value(self.model.optimizer.lr)
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)


# Note: Copy from https://github.com/bckenstler/CLR/blob/master/clr_callback.py
class CyclicLR(LRScheduler):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
        alpha: parameter for exponential moving average
        sma: number of batches for simple moving average
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle', alpha=0.2, sma=20, plot=False, save_plot_prefix=None):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x - 1)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        super(CyclicLR, self).__init__(alpha, sma, plot,
                                       save_plot_prefix + '_cyclic' if save_plot_prefix else None)

    def on_train_begin(self, logs={}):
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.update_lr())

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iteration', []).append(self.clr_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.update_lr()

    def update_lr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)
        K.set_value(self.model.optimizer.lr, lr)


class SGDR(LRScheduler):
    """
    This Callback implements "stochastic gradient descent with warm restarts", a similar approach cyclic approach where
    an cosine annealing schedule is combined with periodic "restarts" to the original starting learning rate.
    See "https://arxiv.org/pdf/1608.03983.pdf" for more details
    """
    def __init__(self, min_lr, max_lr, cycle_length, lr_decay=1, mult_factor=2, alpha=0.2, sma=20, plot=False,
                 save_plot_prefix=None):
        """
        :param min_lr:
        :param max_lr: max_lr and min_lr define the range of desired learning rates
        :param cycle_length: the number of iterations (1 mini-batch is 1 iteration) in an cycle
        :param lr_decay: reduce the max_lr after the completion of each cycle
                         Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        :param mult_factor: scale cycle_length after each full cycle completion
        :param alpha: parameter for exponential moving average
        :param sma: number of batches for simple moving average
        :param plot: whether to plot figures to show the results of the experiment
        :param save_plot_prefix: where to save the figure
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.lr_decay = lr_decay
        self.mult_factor = mult_factor
        self.trn_iterations = 0     # the number of iteration since training
        self.cur_iterations = 0     # the number of iterations since the last restart

        super(SGDR, self).__init__(alpha, sma, plot, save_plot_prefix + '_sgdr' if save_plot_prefix else None)

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.cur_iterations += 1

        self.history.setdefault('iteration', []).append(self.trn_iterations)
        self.history.setdefault('lr', []).append(self.model.optimizer.lr)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.update_lr()

    def update_lr(self):
        fraction_to_restart = self.cur_iterations / self.cycle_length
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(fraction_to_restart * math.pi))
        K.set_value(self.model.optimizer.lr, lr)

        '''Check for end of current cycle'''
        if fraction_to_restart == 1.:
            self.cur_iterations = 0
            self.max_lr = self.lr_decay * self.max_lr
            self.cycle_length = math.ceil(self.cycle_length * self.mult_factor)


# copy from https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452#file-sgdr-py
class SGDRScheduler(LRScheduler):
    """Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    """
    def __init__(self, min_lr, max_lr, steps_per_epoch, lr_decay=1, cycle_length=10, mult_factor=2,
                 alpha=0.2, sma=20, plot=False, save_plot_prefix=None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        super(SGDRScheduler, self).__init__(alpha, sma, plot, save_plot_prefix + '_sgdr2' if save_plot_prefix else None)

    def on_train_begin(self, logs={}):
        # Initialize the learning rate to the minimum value at the start of training.
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        # Record previous batch statistics and update the learning rate.
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1

    def on_epoch_end(self, epoch, logs={}):
        # Check for end of current cycle, apply restarts when necessary.
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay

    def update_lr(self):
        # Calculate the learning rate.
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        K.set_value(self.model.optimizer.lr, lr)


class CyclicLR_1(LRScheduler):
    """
    This Callback implements a cyclic learning schedule introduced in the paper -
       "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs" (https://arxiv.org/pdf/1802.10026.pdf)
    """
    def __init__(self, min_lr, max_lr, cycle_length, alpha=0.2, sma=20, plot=False, save_plot_prefix=None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.trn_iteration = 0
        super(CyclicLR_1, self).__init__(alpha, sma, plot,
                                         save_plot_prefix + '_cyclic_1' if save_plot_prefix else None)

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iteration', []).append(self.trn_iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.update_lr()

    def update_lr(self):
        t = (self.trn_iteration % self.cycle_length) / self.cycle_length
        if t <= 0.5:
            lr = (1 - 2 * t) * self.max_lr + 2 * t * self.min_lr
        else:
            lr = (2 - 2 * t) * self.max_lr + (2 * t - 1) * self.min_lr
        K.set_value(self.model.optimizer.lr, lr)


class CyclicLR_2(LRScheduler):
    """
    This Callback implements a cyclic learning schedule introduced in the paper -
       "AveragingWeights Leads to Wider Optima and Better Generalization" (https://arxiv.org/pdf/1803.05407.pdf)
    """
    def __init__(self, min_lr, max_lr, cycle_length, alpha=0.2, sma=20, plot=False, save_plot_prefix=None):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.trn_iteration = 0
        super(CyclicLR_2, self).__init__(alpha, sma, plot,
                                         save_plot_prefix + '_cyclic_1' if save_plot_prefix else None)

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iteration', []).append(self.trn_iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.update_lr()

    def update_lr(self):
        t = (self.trn_iteration % self.cycle_length) / self.cycle_length
        lr = (1 - t) * self.max_lr + t * self.min_lr
        K.set_value(self.model.optimizer.lr, lr)
