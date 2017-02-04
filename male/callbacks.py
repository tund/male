from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import warnings
import numpy as np
from collections import deque
from .utils.generic_utils import Progbar

import matplotlib.pyplot as plt

plt.style.use('ggplot')


class CallbackList(object):
    def __init__(self, callbacks=[], queue_length=10):
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def _set_params(self, params):
        for callback in self.callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self.callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if self._delta_t_batch > 0. and delta_t_median > 0.95 * \
                self._delta_t_batch and delta_t_median > 0.1:
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch, logs={}):
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if self._delta_t_batch > 0. and (
                        delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)

    def on_train_begin(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    '''Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    '''

    def __init__(self):
        pass

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class BaseLogger(Callback):
    '''Callback that accumulates epoch averages of
    the metrics being monitored.

    This callback is automatically applied to
    every Keras model.
    '''

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in self.totals:
                # make value available to next callbacks
                logs[k] = self.totals[k] / self.seen


class ProgbarLogger(Callback):
    '''Callback that prints metrics to stdout.
    '''

    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.num_epochs = self.params['num_epochs']

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.num_epochs))
            self.progbar = Progbar(target=self.params['num_samples'],
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs={}):
        if self.seen < self.params['num_samples']:
            self.log_values = []

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # skip progbar update for the last batch;
        # will be handled by on_epoch_end
        if self.verbose and self.seen < self.params['num_samples']:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)


class Display(Callback):
    """Callback that visualizes/displays events
    """
    COLOR = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    MARKER = ["x", "d", "o", ">", "<", "^", "v", "|", "1", "s", "+", "*"]
    MARKER_SIZE = 16
    MARKER_EDGE_WIDTH = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    LINESTYLE = ['-', '--', '-.', ':']
    LINE_WIDTH = 4

    def __init__(self, layout=(1, 1), figsize=None, freq=1, monitor=None):
        super(Display, self).__init__()
        self.layout = layout
        self.figsize = figsize
        self.freq = freq
        self.monitor = monitor

    def draw(self, ax, title, **kwargs):
        ax.set_title(title, fontsize=28)
        ax.set_xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else "", fontsize=28)
        ax.set_ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else "", fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=24)

    def on_train_begin(self, logs={}):
        if self.monitor is not None:
            self.fig, self.axs = plt.subplots(
                self.layout[0], self.layout[1], squeeze=False,
                figsize=self.figsize if self.figsize is not None else (
                    12 * self.layout[1], 6.75 * self.layout[0]))
            for i in range(len(self.monitor)):
                u, v = np.unravel_index(i, self.layout, order='C')
                self.draw(self.axs[u, v], **self.monitor[i])
            plt.ion()

    def on_epoch_end(self, epoch, logs={}):
        if ((epoch+1) % self.freq == 0) and (self.monitor is not None):
            for i in range(len(self.monitor)):
                u, v = np.unravel_index(i, self.layout, order='C')
                self.axs[u, v].clear()
                self.draw(self.axs[u, v], **self.monitor[i])
                for (j, value) in enumerate(self.monitor[i]['metrics']):
                    if value in self.model.history_.history:
                        self.disp(self.axs[u, v], j,
                                  np.array(self.model.history_.epoch) + 1,
                                  self.model.history_.history[value],
                                  label=self.monitor[i]['labels'][j] if 'labels' in self.monitor[
                                      i] else value,
                                  **self.monitor[i]
                                  )
                    else:
                        self.model.disp_params(param=value,
                                               epoch=epoch + 1,
                                               ax=self.axs[u, v],
                                               **self.monitor[i])

                if self.axs[u, v].get_legend() is None:
                    self.axs[u, v].legend(fontsize=24, numpoints=1)
            plt.pause(0.0001)
            self.fig.tight_layout()

    def disp(self, ax, id, x, y, type, label, *args, **kwargs):
        if type == 'line':
            _ = ax.plot(x, y, label=label,
                        color=kwargs['color'] if 'color' in kwargs else self.COLOR[
                            id % len(self.COLOR)],
                        linestyle=kwargs['linestyle'] if 'linestyle' in kwargs else self.LINESTYLE[
                            id % len(self.LINESTYLE)],
                        linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else self.LINE_WIDTH,
                        marker=kwargs['marker'] if 'marker' in kwargs else self.MARKER[
                            id % len(self.MARKER)],
                        markersize=kwargs[
                            'markersize'] if 'markersize' in kwargs else self.MARKER_SIZE,
                        markeredgewidth=kwargs[
                            'markeredgewidth'] if 'markeredgewidth' in kwargs else
                        self.MARKER_EDGE_WIDTH[id % len(self.MARKER_EDGE_WIDTH)],
                        markevery=len(x) // 10 + 1,
                        )
        elif type == 'img':
            _ = ax.imshow(x, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')

    def on_train_end(self, logs={}):
        plt.show(block=True)


class History(Callback):
    '''Callback that records events
    into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    '''

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class ModelCheckpoint(Callback):
    '''Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.

    '''

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class EarlyStopping(Callback):
    '''Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    '''

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0,
                 mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs={}):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_idx = 0  # index of the epoch with the best performance

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.best_idx = epoch + 1
            self.model.best_ = current
            self.model.best_params_ = self.model.get_all_params()
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.set_params(**self.model.best_params_)
                self.model.stop_training_ = self.best_idx
            self.wait += 1

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))
