from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
import os
import sys
import time
import warnings

import dill as pkl
import numpy as np
from scipy.optimize import check_grad as scipy_check_grad
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_random_state

from .configs import model_dir
from . import callbacks as cbks
from .utils.generic_utils import tuid
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
                 catch_exception=False,
                 random_state=None,
                 verbose=0):
        self.model_name = model_name
        self.task = task
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cv = cv
        self.callbacks = callbacks
        self.metrics = metrics
        self.catch_exception = catch_exception
        self.random_state = random_state
        self.verbose = verbose

    def _init(self):
        self.epoch = 0
        self.history = None
        self.num_classes = 0
        self.label_encoder = None
        self.start_time = 0.0
        self.train_time = 0.0
        self.stop_training = 0
        self.exception = False
        self.best = -INF
        self.best_params = None
        self.random_engine = check_random_state(self.random_state)

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

        if (not hasattr(self, 'epoch')) or self.epoch == 0:
            self._init()
            if y_train is not None:
                # encode labels
                y_train = self._encode_labels(y_train)
            self._init_params(x_train)
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
            'batch_size': self.batch_size,
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
                               callbacks=callbacks, callback_metrics=callback_metrics)
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
                           callbacks=callbacks, callback_metrics=callback_metrics)

        callbacks.on_train_end()
        self._on_train_end()

        return self

    def _fit_loop(self, x, y, *args, **kwargs):
        pass

    def _on_train_begin(self):
        self.stop_training = 0
        self.start_time = time.time()

    def _on_train_end(self):
        self.train_time = time.time() - self.start_time

    def _on_epoch_begin(self):
        pass

    def _on_epoch_end(self):
        self.epoch += 1
        if self.stop_training:
            self.epoch = self.stop_training

    def _on_batch_begin(self, x, y=None):
        pass

    def _on_batch_end(self, x, y=None):
        outs = {}
        for m in self.metrics:
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
            self.label_encoder = LabelEncoder()
            yy = self.label_encoder.fit_transform(yy)
            self.num_classes = len(self.label_encoder.classes_)
        elif self.task == 'multilabel':
            self.label_encoder = MultiLabelBinarizer()
            yy = self.label_encoder.fit_transform(yy)
            self.num_classes = len(self.label_encoder.classes_)
        return yy

    def _decode_labels(self, y):
        if self.task == 'classification' or self.task == 'multilabel':
            return self.label_encoder.inverse_transform(y)
        else:
            return y

    def _transform_labels(self, y):
        if self.task == 'classification' or self.task == 'multilabel':
            return self.label_encoder.transform(y)
        else:
            return y

    def _roll_params(self):
        return []

    def _unroll_params(self, w):
        return ()

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass

    def score(self, x, y, sample_weight=None):
        if self.exception:
            return -INF
        else:
            if self.task == 'classification':
                return float(accuracy_score(y, self.predict(x)))
            elif self.task == 'multilabel':
                return float(f1_score(self._transform_labels(y),
                                      self._transform_labels(self.predict(x)),
                                      average='weighted'))
            else:
                return -float(mean_squared_error(y, self.predict(x)))

    def save(self, file_path=None, overwrite=True):
        if file_path is None:
            file_path = os.path.join(model_dir(), "male/{}/{}.pkl".format(self.model_name, tuid()))
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
        pkl.dump({'model': self}, open(file_path, 'wb'))

    @staticmethod
    def load_model(file_path):
        return pkl.load(open(file_path, 'rb'))['model']

    def display(self, param, **kwargs):
        pass

    @abc.abstractmethod
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        param_names = Model._get_param_names()
        return self._get_params(param_names=param_names, deep=deep)

    def _get_params(self, param_names=[], deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in param_names:
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def get_all_params(self):
        return self.__dict__

    def set_all_params(self, **params):
        valid_params = self.__dict__.keys()
        for p, value in params.items():
            if p in valid_params:
                setattr(self, p, value)
            else:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.__dict__.keys()`.' %
                                 (p, self.__class__.__name__))
        return self

    def __setstate__(self, state):
        from . import __version__
        if type(self).__module__.startswith('male.'):
            pickle_version = state.pop("_male_version", "0.1.0")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        self.__dict__.update(state)
