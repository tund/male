from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import abc
import sys
import copy
import time
import numpy as np
import pickle as pkl

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import check_grad as scipy_check_grad

from . import callbacks as cbks
from .utils.io_utils import ask_to_proceed_with_overwrite

INF = 1e+8


class Model(BaseEstimator, ClassifierMixin,
            RegressorMixin, TransformerMixin):
    """A generic class of model
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model_name='male',
                 task='classification',
                 num_epochs=10,
                 batch_size=32,
                 cv=None,  # cross-validation
                 callbacks=[],
                 metrics=[],  # {'loss', 'acc', 'err'}
                 random_state=None,
                 verbose=0):
        self.model_name = model_name
        self.task = task
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cv = cv
        self.callbacks = callbacks
        self.metrics = metrics
        self.random_state = random_state
        self.verbose = verbose

    def _init(self):
        self.epoch_ = 0
        self.history_ = None
        self.num_classes_ = 0
        self.label_encoder_ = None
        self.train_time_ = 0.0
        self.stop_training_ = 0
        self.exception_ = False
        self.best_ = -INF
        self.best_params_ = None
        self.random_state_ = check_random_state(self.random_state)

    def _init_params(self, x):
        pass

    def fit(self, x=None, y=None):
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

        if (not hasattr(self, 'epoch_')) or self.epoch_ == 0:
            self._init()
            if y_train is not None:
                # encode labels
                y_train = self._encode_labels(y_train)
            self._init_params(x_train)
        else:  # continue training
            if y_train is not None:
                y_train = self._transform_labels(y_train)

        self.history_ = cbks.History()
        callbacks = [cbks.BaseLogger()] + self.callbacks + [self.history_]
        if self.verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        if not do_validation:
            callback_metrics = copy.copy(self.metrics)
        else:
            callback_metrics = (copy.copy(self.metrics) + ['val_' + m for m in self.metrics])

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'num_samples': x_train.shape[0] if x_train is not None else 0,
            'verbose': self.verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })

        callbacks.on_train_begin()
        self.stop_training_ = 0

        try:
            start_time = time.time()
            self._fit_loop(x_train, y_train,
                           do_validation=do_validation,
                           x_valid=x_valid, y_valid=y_valid,
                           callbacks=callbacks, callback_metrics=callback_metrics)
            self.train_time_ = time.time() - start_time
        except KeyboardInterrupt:
            sys.exit()
        except:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
            self._init_params(x_train)  # reset all parameters
            self.exception_ = True
            return self

        callbacks.on_train_end()

        return self

    def _fit_loop(self, x, y, *args, **kwargs):
        pass

    def _on_batch_end(self, x, y=None):
        outs = []
        for m in self.metrics:
            if m == 'loss':
                outs += [self.get_loss(x, y)]
            if m == 'acc':
                outs += [self.score(x, self._decode_labels(y))]
            if m == 'err':
                if self.task == 'classification':
                    outs += [1 - self.score(x, self._decode_labels(y))]
                else:
                    outs += [-self.score(x, self._decode_labels(y))]
        return outs

    def check_grad(self, x, y=None):
        """Check gradients of the model using data x and label y if available
        """
        self._init()

        if y is not None:
            # encode labels
            y = self._encode_labels(y)

        # initialize weights
        self._init_params(x)

        print("Checking gradient... ", end='')
        diff = scipy_check_grad(self._get_loss_check_grad,
                                self._get_grad_check_grad,
                                self._roll_params(), x, y)

        print("diff = %.8f" % diff)
        return diff

    def get_loss(self, x, y, *args, **kwargs):
        pass

    def _get_loss_check_grad(self, w, x, y):
        pass

    def get_grad(self, x, y, *args, **kwargs):
        pass

    def _get_grad_check_grad(self, w, x, y):
        pass

    def _encode_labels(self, y):
        yy = y.copy()
        if self.task == 'classification':
            self.label_encoder_ = LabelEncoder()
            yy = self.label_encoder_.fit_transform(yy)
            self.num_classes_ = len(self.label_encoder_.classes_)
        return yy

    def _decode_labels(self, y):
        if self.task == 'classification':
            return self.label_encoder_.inverse_transform(y)
        else:
            return y

    def _transform_labels(self, y):
        if self.task == 'classification':
            return self.label_encoder_.transform(y)
        else:
            return y

    def _roll_params(self):
        return []

    def _unroll_params(self, w):
        return ()

    def predict(self, x):
        pass

    def score(self, x, y, sample_weight=None):
        if self.exception_:
            return -INF
        else:
            if self.task == 'classification':
                return float(accuracy_score(self.predict(x), y))
            else:
                return -float(mean_squared_error(self.predict(x), y))

    def save(self, file_path, overwrite=True):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        self._save_model(file_path, overwrite)

    def _save_model(self, file_path, overwrite=True):
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(file_path):
            proceed = ask_to_proceed_with_overwrite(file_path)
            if not proceed:
                return
        pkl.dump({'model': self}, open(file_path, 'wb'))

    def _load_model(self, file_path):
        return pkl.load(open(file_path, 'rb'))['model']

    def set_params(self, **params):
        for p, value in params.items():
            setattr(self, p, value)
        return self

    @abc.abstractmethod
    def get_params(self, deep=True):
        return {'model_name': self.model_name,
                'task': self.task,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'cv': self.cv,
                'callbacks': self.callbacks,
                'metrics': self.metrics,
                'random_state': self.random_state,
                'verbose': self.verbose}

    @abc.abstractmethod
    def get_all_params(self, deep=True):
        out = self.get_params(deep=deep)
        out.update({'epoch_': self.epoch_,
                    'train_time_': self.train_time_,
                    'num_classes_': self.num_classes_,
                    'best_': self.best_,
                    'stop_training_': self.stop_training_,
                    'history_': copy.deepcopy(self.history_),
                    'label_encoder_': copy.deepcopy(self.label_encoder_),
                    'random_state_': copy.deepcopy(self.random_state)})
        return out
