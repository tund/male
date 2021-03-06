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
from .configs import model_dir
from . import callbacks as cbks
from .backend import tensorflow as tf_backend
from .utils.generic_utils import tuid
from .utils.io_utils import ask_to_proceed_with_overwrite


class TensorFlowModel(Model):
    def __init__(self,
                 model_name="TensorFlowMale",
                 log_path=None,
                 model_path=None,
                 summary_freq=int(1e+8),
                 per_process_gpu_memory_fraction=None,
                 **kwargs):
        super(TensorFlowModel, self).__init__(model_name=model_name, **kwargs)
        self.log_path = log_path
        self.model_path = model_path
        self.summary_freq = summary_freq
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

    def _init(self):
        super(TensorFlowModel, self)._init()

        self.tf_graph = tf.Graph()
        self.tf_config = tf_backend.get_default_config(
            per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)
        self.tf_merged_summaries = None
        self.tf_summary_writer = None

        if self.random_state is not None:
            with self.tf_graph.as_default():
                tf.set_random_seed(self.random_state)

        # create logging directory
        if self.log_path is None:
            self.log_path = os.path.join(model_dir(), self.model_name, "logs",
                                         "{}".format(tuid()))
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))

    def _get_session(self, **kwargs):
        graph = kwargs['graph'] if 'graph' in kwargs else self.tf_graph
        sess = self.tf_session if 'sess' not in kwargs else kwargs['sess']
        # if sess is None:
        #     sess = tf.Session(config=self.tf_config, graph=graph)
        #     self.tf_saver.restore(sess, self.model_path)
        return sess

    def _build_model(self, x):
        pass

    def fit(self, x=None, y=None, **kwargs):
        """Fit the model to the data X and the label y if available
        """

        # copy to avoid modifying
        x = copy.deepcopy(x) if x is not None else x
        y = copy.deepcopy(y) if y is not None else y

        if (x is not None) and (hasattr(x, 'ndim')) and (x.ndim < 2):
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
            if (x_train is not None) and (hasattr(x_train, 'shape')):
                self.data_dim = x_train.shape[1]
            if y_train is not None:
                # encode labels
                y_train = self._encode_labels(y_train)
            with self.tf_graph.as_default():
                self._init_params(x_train)
                self._build_model(x_train)
                # merge all summaries
                self.tf_merged_summaries = tf.summary.merge_all()
                # Initialize all variables
                self.tf_session.run(tf.global_variables_initializer())
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
            'num_samples': x_train.shape[0]
            if (x_train is not None) and (hasattr(x_train, 'shape'))
            else self.batch_size,
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

    def _on_train_begin(self):
        super(TensorFlowModel, self)._on_train_begin()
        # summary writer
        self.tf_summary_writer = tf.summary.FileWriter(self.log_path, self.tf_session.graph)

    def _on_train_end(self):
        super(TensorFlowModel, self)._on_train_end()
        self.tf_summary_writer.close()

    def _on_epoch_end(self, **kwargs):
        super(TensorFlowModel, self)._on_epoch_end()
        if self.epoch % self.summary_freq == 0:
            if "input_data" in kwargs:
                _summary = self.tf_session.run(self.tf_merged_summaries,
                                               feed_dict=kwargs["input_data"])
            else:
                _summary = self.tf_session.run(self.tf_merged_summaries)
            self.tf_summary_writer.add_summary(_summary, self.epoch)

    def save(self, file_path=None, overwrite=True):
        if file_path is None:
            file_path = os.path.join(model_dir(),
                                     "male/{}/{}.ckpt".format(self.model_name, tuid()))
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

    def _on_load_model_pickle_end(self):
        pass

    @staticmethod
    def load_model(file_path):
        model = pkl.load(open(file_path + ".pkl", 'rb'))['model']
        model._on_load_model_pickle_end()
        if not hasattr(model, 'per_process_gpu_memory_fraction'):
            model.per_process_gpu_memory_fraction = None  # default
        model.tf_graph = tf.Graph()
        model.tf_config = tf_backend.get_default_config(
            per_process_gpu_memory_fraction=model.per_process_gpu_memory_fraction)
        model.tf_session = tf.Session(config=model.tf_config, graph=model.tf_graph)
        with model.tf_graph.as_default():
            # tf.get_variable_scope().reuse_variables()
            model._init_params(None)
            model._build_model(None)
            # merge all summaries
            model.tf_merged_summaries = tf.summary.merge_all()
            saver = tf.train.Saver()
            saver.restore(model.tf_session, file_path)
        model.best_params = model.get_all_params()
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
        # param_names = ['tf_graph', 'tf_config', 'tf_merged_summaries']
        # for key in param_names:
        #     if key in self.__dict__:
        #         out[key] = self.__getattribute__(key)
        return out

    def __getstate__(self):
        from . import __version__
        out = self._get_params_to_dump(deep=True)
        if type(self).__module__.startswith('male.'):
            return dict(out, _male_version=__version__)
        else:
            return out
