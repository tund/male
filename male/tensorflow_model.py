from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import six
import copy
import numpy as np
import dill as pkl
import tensorflow as tf

from . import Model
from . import model_dir
from . import callbacks as cbks
from .backend import tensorflow_backend as tf_backend
from .utils.generic_utils import tuid
from .utils.io_utils import ask_to_proceed_with_overwrite


class TensorFlowModel(Model):
    def __init__(self,
                 model_name="TensorFlowMale",
                 model_path=None,
                 **kwargs):
        super(TensorFlowModel, self).__init__(model_name=model_name, **kwargs)
        self.model_path = model_path

    def _init(self):
        super(TensorFlowModel, self)._init()

        self.tf_graph = tf.Graph()
        self.tf_config = tf_backend.get_default_config()
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)
        self.tf_merged_summaries = None
        self.tf_summary_writer = None

        if self.random_state is not None:
            with self.tf_graph.as_default():
                tf.set_random_seed(self.random_state)

    def _get_session(self, **kwargs):
        graph = kwargs['graph'] if 'graph' in kwargs else self.tf_graph
        sess = self.tf_session if 'sess' not in kwargs else kwargs['sess']
        # if sess is None:
        #     sess = tf.Session(config=self.tf_config, graph=graph)
        #     self.tf_saver.restore(sess, self.model_path)
        return sess

    def _build_model(self, x):
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
            if x_train is not None:
                self.data_dim = x_train.shape[1]
            if y_train is not None:
                # encode labels
                y_train = self._encode_labels(y_train)
            with self.tf_graph.as_default():
                self._init_params(x_train)
                self._build_model(x_train)
                # Initialize all variables
                self.tf_session.run(tf.global_variables_initializer())
        else:  # continue training
            if y_train is not None:
                y_train = self._transform_labels(y_train)

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

        return self.history

    def save(self, file_path=None, overwrite=True):
        if file_path is None:
            file_path = os.path.join(model_dir(), self.model_name,
                                     "{}_{}.ckpt".format(tuid(), self.model_name))
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
        pkl.dump({'model': self}, open(file_path + ".pkl", 'wb'))
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.tf_session, file_path)

    @staticmethod
    def load_model(file_path):
        model = pkl.load(open(file_path + ".pkl", 'rb'))['model']
        model.tf_session = tf.Session(config=model.tf_config, graph=model.tf_graph)
        with model.tf_graph.as_default():
            tf.get_variable_scope().reuse_variables()
            model._init_params(None)
            model._build_model(None)
            saver = tf.train.Saver()
            saver.restore(model.tf_session, file_path)
        return model

    def get_params(self, deep=True):
        out = super(TensorFlowModel, self).get_params(deep=deep)
        param_names = TensorFlowModel._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out

    def _get_params_to_dump(self, deep=True):
        out = dict()
        for key, value in six.iteritems(self.__dict__):
            if ((not type(value).__module__.startswith('tf')) and
                    (not type(value).__module__.startswith('tensorflow')) and
                    (key != 'best_params')):
                out[key] = value
        param_names = ['tf_graph', 'tf_config', 'tf_merged_summaries', 'tf_summary_writer']
        for key in param_names:
            if key in self.__dict__:
                out[key] = self.__getattribute__(key)
        return out

    def __getstate__(self):
        from . import __version__
        out = self._get_params_to_dump(deep=True)
        if type(self).__module__.startswith('male.'):
            return dict(out, _male_version=__version__)
        else:
            return dict(self.__dict__.items())
