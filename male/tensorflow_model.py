from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import copy
import numpy as np
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
tf_config.allow_soft_placement = True

from . import Model
from . import callbacks as cbks
from .utils.io_utils import ask_to_proceed_with_overwrite


class TensorFlowModel(Model):
    def __init__(self,
                 model_name="TensorFlowMale",
                 model_path="rmodel/male",
                 **kwargs):
        kwargs["model_name"] = model_name
        super(TensorFlowModel, self).__init__(**kwargs)
        from . import HOME
        self.model_path = os.path.join(HOME, model_path + "/" + model_name + "/" + model_name)

    def _init(self):
        super(TensorFlowModel, self)._init()

        self.tf_graph_ = tf.Graph()
        self.tf_session_ = tf.Session(config=tf_config, graph=self.tf_graph_)
        self.tf_saver_ = None
        self.tf_merged_summaries_ = None
        self.tf_summary_writer_ = None

        if self.random_state is not None:
            tf.set_random_seed(self.random_state)

    def _get_session(self, **kwargs):
        graph = kwargs['graph'] if 'graph' in kwargs else self.tf_graph_
        sess = self.tf_session_ if 'sess' not in kwargs else kwargs['sess']
        if sess is None:
            sess = tf.Session(config=tf_config, graph=graph)
            self.tf_saver_.restore(sess, self.model_path)
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

        if (not hasattr(self, 'epoch_')) or self.epoch_ == 0:
            self._init()
            if y_train is not None:
                # encode labels
                y_train = self._encode_labels(y_train)
            with self.tf_graph_.as_default():
                self._init_params(x_train)
                self._build_model(x_train)
        else:  # continue training
            if y_train is not None:
                y_train = self._transform_labels(y_train)

        self.history_ = cbks.History()
        callbacks = [cbks.BaseLogger()] + [self.history_] + self.callbacks
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
                self.exception_ = True
                return self
        else:
            self._fit_loop(x_train, y_train,
                           do_validation=do_validation,
                           x_valid=x_valid, y_valid=y_valid,
                           callbacks=callbacks, callback_metrics=callback_metrics)

        callbacks.on_train_end()
        self._on_train_end()

        return self

    def _on_train_begin(self):
        super(TensorFlowModel, self)._on_train_begin()
        with self.tf_graph_.as_default():
            self.tf_saver_ = tf.train.Saver()

    def _on_train_end(self):
        super(TensorFlowModel, self)._on_train_end()
        self.save(file_path=self.model_path)
        self.tf_session_.close()
        self.tf_session_ = None

    def _save_model(self, file_path, overwrite=True):
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(file_path):
            proceed = ask_to_proceed_with_overwrite(file_path)
            if not proceed:
                return
        # pkl.dump({'model': self}, open(file_path, 'wb'))
        self.tf_saver_.save(self.tf_session_, file_path)

    def get_all_params(self, deep=True):
        out = super(TensorFlowModel, self).get_all_params(deep=deep)
        out.update(self.get_params(deep=deep))
        out.update({
            'tf_graph_': self.tf_graph_,
            'tf_session_': self.tf_session_,
            'tf_saver_': self.tf_saver_,
            'tf_merged_summaries_': self.tf_merged_summaries_,
            'tf_summary_writer_': self.tf_summary_writer_,
        })
        return out
