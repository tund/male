from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import six
import copy
import torch
from torch import nn
import numpy as np
import dill as pkl

from . import Model
from .configs import model_dir
from . import callbacks as cbks
from .utils.generic_utils import tuid
from sklearn.utils.validation import check_is_fitted
from .utils.io_utils import ask_to_proceed_with_overwrite


class PyTorchModel(Model):
    def __init__(self,
                 model_name='PyTorchMale',
                 float=torch.float32,
                 default_device='cpu',
                 log_path=None,
                 model_path=None,
                 summary_freq=int(1e+8),
                 **kwargs):
        super(PyTorchModel, self).__init__(model_name=model_name, **kwargs)
        self.float = float
        self.default_device = default_device
        self.log_path = log_path
        self.model_path = model_path
        self.summary_freq = summary_freq

    def _init(self):
        super(PyTorchModel, self)._init()

        # create logging directory
        if self.log_path is None:
            self.log_path = os.path.join(model_dir(), self.model_name, "logs",
                                         "{}".format(tuid()))
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))

        # init device
        if 'cuda' in self.default_device and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(self.default_device)

    def _build_model(self, x):
        self.net = None
        pass

    def fit(self, x=None, y=None, **kwargs):
        """Fit the model to the data X and the label y if available
        """

        # copy to avoid modifying
        x = x.copy() if x is not None else x
        y = y.copy() if y is not None else y

        if (x is not None) and (x.ndim < 2):
            x = x[..., np.newaxis]

        if self.cv is not None:
            do_validation = True
            idx = np.array(self.cv)
            x_train = x[idx < 0]
            y_train = y[idx < 0] if y is not None else None
            x_valid = x[idx >= 0]
            y_valid = y[idx >= 0] if y is not None else None
            if self.verbose:
                print('Train on %d samples, validate on %d samples' %
                      (x_train.shape[0], x_valid.shape[0]))
        else:
            do_validation = False
            x_train, y_train = x, y
            x_valid, y_valid = None, None

        if (not hasattr(self, 'epoch')) or self.epoch == 0:
            self._init()
            if y_train is not None:
                # encode labels
                y_train = self._encode_labels(y_train)
            self._init_params(x_train)
            self._build_model(x_train)
        else:  # continue training
            if y_train is not None:
                y_train = self._transform_labels(y_train)

        if (not hasattr(self, 'history')) or self.history is None:
            self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + [self.history] + self.callbacks
        if self.verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        if not do_validation:
            callback_metrics = copy.copy(self.metrics)
        else:
            callback_metrics = (copy.copy(self.metrics) + ['val_' + m for m in self.metrics])

        callbacks._set_model(self)
        callbacks._set_params({
            'num_epochs': self.num_epochs,
            'num_samples': x_train.shape[0] if x_train is not None else self.batch_size,
            'verbose': self.verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })

        self._on_train_begin()
        callbacks.on_train_begin()

        if self.catch_exception:
            try:
                self._fit_loop(x_train, y_train,
                               do_validation=do_validation,
                               x_valid=x_valid, y_valid=y_valid,
                               callbacks=callbacks, callback_metrics=callback_metrics, **kwargs)
            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Unexpected error: {}".format(sys.exc_info()[0]))
                self._init_params(x_train)  # reset all parameters
                self.exception = True
                return self
        else:
            self._fit_loop(x_train, y_train,
                           do_validation=do_validation,
                           x_valid=x_valid, y_valid=y_valid,
                           callbacks=callbacks, callback_metrics=callback_metrics, **kwargs)

        callbacks.on_train_end()
        self._on_train_end()

        return self

    def forward(self, x):
        return self.net(x)

    def get_loss(self, x, y, **kwargs):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).type(self.float).to(self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y).to(self.device)
            y_pred = self.forward(x)
            return self.criterion(y_pred, y).item()

    def _on_batch_end(self, x, y=None, logs={}):
        outs = {}
        for m in self.metrics:
            if m in logs:
                continue
            if m == 'loss':
                outs.update({m: self.get_loss(x, y)})
            if m == 'acc':
                outs.update({m: self.score(x, self._decode_labels(y))})
            if m == 'err':
                if self.task == 'classification':
                    outs.update({m: 1 - self.score(x, self._decode_labels(y))})
                else:
                    outs.update({m: -self.score(x, self._decode_labels(y))})
        return outs

    def save(self, file_path=None, overwrite=True):
        if file_path is None:
            file_path = os.path.join(model_dir(),
                                     'male/{}/{}.pth'.format(self.model_name, tuid()))
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        self._save_model(file_path, overwrite)
        return file_path

    def _save_model(self, file_path, overwrite=True):
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(file_path):
            proceed = ask_to_proceed_with_overwrite(file_path)
            if not proceed:
                return
        # torch.save(self, file_path)
        pkl.dump({'model': self}, open(file_path, 'wb'))

    @staticmethod
    def load_model(file_path):
        # model = torch.load(file_path)
        model = pkl.load(open(file_path, 'rb'))['model']        
        model.net.eval() 
        return model

    def get_params(self, deep=True):
        out = super(PyTorchModel, self).get_params(deep=deep)
        param_names = PyTorchModel._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out

    def _get_params_to_dump(self, deep=True):
        out = dict()
        for key, value in six.iteritems(self.__dict__):
            if key != 'best_params':
                out[key] = value
        return out

    def __getstate__(self):
        from . import __version__
        out = self._get_params_to_dump(deep=True)
        if type(self).__module__.startswith('male.'):
            return dict(out, _male_version=__version__)
        else:
            return out
