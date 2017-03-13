from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import abc
import copy

import numpy as np
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
tf_config.allow_soft_placement = True

from sklearn.utils.validation import check_is_fitted

from .rbm import RBM
from ....tensorflow_model import TensorFlowModel
from ....utils.generic_utils import make_batches
from ....utils.disp_utils import tile_raster_images
from ....utils.func_utils import logsumone

import matplotlib.pyplot as plt

plt.style.use('ggplot')

LEARNING_METHOD = {'cd': 0, 'pcd': 1}
MOMENTUM_METHOD = {'none': 0, 'sudden': 1}
INFERENCE_ENGINE = {'gibbs': 0, 'variational_inference': 1}
DECAY_METHOD = {'none': 0, 'linear': 1, 'div_sqrt': 2, 'exp': 3}
CD_SAMPLING = {'hidden_visible': 0, 'hidden': 1, 'visible': 2, 'none': 3}


class TensorFlowRBM(TensorFlowModel, RBM):
    """A generic class of Restricted Boltzmann Machine using TensorFlow
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model_name="TensorFlowRBM",
                 *args, **kwargs):

        kwargs["model_name"] = model_name
        super(TensorFlowRBM, self).__init__(**kwargs)

    def _init(self):
        super(TensorFlowRBM, self)._init()
        with self.tf_graph_.as_default():
            self.tf_learning_rate_ = tf.get_variable(
                "learning_rate", shape=[],
                initializer=tf.constant_initializer(self.learning_rate))
            self.tf_momentum_ = tf.get_variable(
                "momentum", shape=[],
                initializer=tf.constant_initializer(self.initial_momentum))

    def _init_params(self, x):
        # initialize parameters
        self.tf_h_ = tf.get_variable("hidden_bias", shape=[1, self.num_hidden],
                                     initializer=tf.random_normal_initializer(stddev=self.h_init))
        self.tf_v_ = tf.get_variable("visible_bias", shape=[1, self.num_visible],
                                     initializer=tf.random_normal_initializer(stddev=self.v_init))
        self.tf_w_ = tf.get_variable("weight", shape=[self.num_visible, self.num_hidden],
                                     initializer=tf.random_normal_initializer(stddev=self.w_init))
        self.tf_hgrad_inc_ = tf.get_variable("hidden_grad_inc", shape=[1, self.num_hidden],
                                             initializer=tf.constant_initializer(0.0))
        self.tf_vgrad_inc_ = tf.get_variable("visible_grad_inc", shape=[1, self.num_visible],
                                             initializer=tf.constant_initializer(0.0))
        self.tf_wgrad_inc_ = tf.get_variable("weight_grad_inc",
                                             shape=[self.num_visible, self.num_hidden],
                                             initializer=tf.constant_initializer(0.0))

    def _build_model(self, x):
        self.tf_x_ = tf.placeholder(tf.float32, shape=[None, self.num_visible], name="data")

        self.tf_epoch_ = tf.placeholder(tf.float32, shape=[], name="epoch")
        # self.hidden_prob_ = self._get_hidden_prob(self.tf_x_)
        self.tf_hidden_prob_ = self._create_hidden_prob(self.tf_x_)
        self.tf_reconstruction_ = self._create_reconstruction(self.tf_x_)
        self.tf_reconstruction_error_ = self._create_reconstruction_error(self.tf_x_,
                                                                          tf_rdata=self.tf_reconstruction_)
        self.tf_free_energy_ = self._create_free_energy(self.tf_x_)

        self.tf_reconstruction_loglik_ = self._create_reconstruction_loglik(self.tf_x_,
                                                                            tf_rdata=self.tf_reconstruction_)

        tf_pos_hgrad, tf_pos_vgrad, tf_pos_wgrad = self._initialize_grad()

        tf_prev_hprob = tf.get_variable("previous_hidden_prob",
                                        shape=[self.batch_size, self.num_hidden],
                                        initializer=tf.constant_initializer(0.0))

        # ======= clamp phase ========
        # hprob = self._get_hidden_prob(self.x_)
        tf_hprob = self._create_hidden_prob(self.tf_x_)

        # sparsity
        if self.sparse_weight > 0:
            tf_hg, tf_wg, tf_prev_hprob = self._create_hidden_sparsity(self.tf_x_, tf_prev_hprob,
                                                                       tf_hprob)
            tf_pos_hgrad += tf_hg
            tf_pos_wgrad += tf_wg

        tf_hg, tf_vg, tf_wg = self._create_positive_grad(self.tf_x_, tf_hprob)
        tf_pos_hgrad += tf_hg
        tf_pos_vgrad += tf_vg
        tf_pos_wgrad += tf_wg

        # ======== free phase =========
        if self.learning_method == LEARNING_METHOD['cd']:
            for icd in range(self.num_cd - 1):
                tf_hsample, tf_vprob, tf_vsample, tf_hprob = self._create_gibbs_sampling(
                    tf_hprob, sampling=CD_SAMPLING['hidden_visible'])
            tf_hsample, tf_vprob, tf_vsample, tf_hprob = self._create_gibbs_sampling(
                tf_hprob, sampling=self.sampling_in_last_cd)

        # ======== negative phase =========
        tf_neg_hgrad, tf_neg_vgrad, tf_neg_wgrad = self._create_negative_grad(tf_vprob, tf_hprob)

        self.tf_hgrad_inc_ = self.tf_momentum_ * self.tf_hgrad_inc_ + self.tf_learning_rate_ * (
            tf_pos_hgrad - tf_neg_hgrad)
        self.tf_vgrad_inc_ = self.tf_momentum_ * self.tf_vgrad_inc_ + self.tf_learning_rate_ * (
            tf_pos_vgrad - tf_neg_vgrad)
        self.tf_wgrad_inc_ = self.tf_momentum_ * self.tf_wgrad_inc_ \
                             + self.tf_learning_rate_ * (
            tf_pos_wgrad - tf_neg_wgrad - self.weight_cost * self.tf_w_)

        # update params
        self.tf_h_update_ = self.tf_h_.assign_add(self.tf_hgrad_inc_)
        self.tf_v_update_ = self.tf_v_.assign_add(self.tf_vgrad_inc_)
        self.tf_w_update_ = self.tf_w_.assign_add(self.tf_wgrad_inc_)

        # update learning rate and momentum
        self.tf_learning_rate_update_ = self._create_learning_rate(self.tf_epoch_)
        self.tf_momentum_update_ = self._create_momentum(self.tf_epoch_)

        self.tf_session_.run(tf.global_variables_initializer())

    @abc.abstractmethod
    def transform(self, x):
        """Compute the hidden layer probabilities, P(h|v=X).

        Parameters
        ----------
        x : {array-like, sparse matrix} shape (num_samples, num_visible)
            The data to be transformed.

        Returns
        -------
        h : array, shape (num_samples, num_hidden)
            Latent representations of the data.
        """
        check_is_fitted(self, "w_")
        return None

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        """Fit the model to the data X

        Parameters
        ----------
        x : {array-like, sparse matrix} shape (num_samples, num_visible)
            Training data.

        Returns
        -------
        self : RBM
            The fitted model.
        """

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch_ < self.num_epochs) and (not self.stop_training_):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch_)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]

                self.tf_session_.run(
                    [self.tf_w_update_, self.tf_h_update_, self.tf_v_update_],
                    feed_dict={self.tf_x_: x_batch})
                # self.v_, self.h_, self.w_, self.vgrad_inc_, self.hgrad_inc_, self.wgrad_inc_ \
                #     = self._get_params_from_tf_params()
                self.v_, self.h_, self.w_ = self._get_params_from_tf_params()
                batch_logs.update(self._on_batch_end(x_batch))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid)
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch_, epoch_logs)
            self._on_epoch_end()

    # def _gibbs_sampling(self, hprob, sampling=CD_SAMPLING['hidden_visible']):

    def _create_gibbs_sampling(self, tf_hprob, sampling=CD_SAMPLING['hidden_visible']):
        if sampling == CD_SAMPLING['hidden']:
            tf_hsample = self._create_sample_hidden(tf_hprob)
            tf_vprob = self._create_visible_prob(tf_hsample)
            tf_vsample = tf.identity(tf_vprob)  # copy
            tf_hprob = self._create_hidden_prob(tf_vsample)
        elif sampling == CD_SAMPLING['none']:
            tf_hsample = tf.identity(tf_hprob)  # copy
            tf_vprob = self._create_visible_prob(tf_hsample)
            tf_vsample = tf.identity(tf_vprob)  # copy
            tf_hprob = self._create_hidden_prob(tf_vsample)
        elif sampling == CD_SAMPLING['hidden_visible']:
            tf_hsample = self._create_sample_hidden(tf_hprob)
            tf_vprob = self._create_visible_prob(tf_hsample)
            tf_vsample = self._create_sample_visible(tf_vprob)
            tf_hprob = self._create_hidden_prob(tf_vsample)
        else:
            tf_hsample = tf.identity(tf_hprob)  # copy
            tf_vprob = self._create_visible_prob(tf_hsample)
            tf_vsample = self._create_sample_visible(tf_vprob)
            tf_hprob = self._create_hidden_prob(tf_vsample)

        return tf_hsample, tf_vprob, tf_vsample, tf_hprob

    def _initialize_grad(self):
        tf_pos_hgrad = tf.get_variable("positive_hidden_grad", shape=[1, self.num_hidden],
                                       initializer=tf.constant_initializer(0.0))
        tf_pos_vgrad = tf.get_variable("positive_visible_grad", shape=[1, self.num_visible],
                                       initializer=tf.constant_initializer(0.0))
        tf_pos_wgrad = tf.get_variable("positive_weight_grad",
                                       shape=[self.num_visible, self.num_hidden],
                                       initializer=tf.constant_initializer(0.0))
        return tf_pos_hgrad, tf_pos_vgrad, tf_pos_wgrad

    def _create_hidden_sparsity(self, tf_x, tf_prev_hprob, tf_hprob):
        if tf_prev_hprob.get_shape() == tf_hprob.get_shape():
            q = self.sparse_decay * tf_prev_hprob + (1 - self.sparse_decay) * tf_hprob
        else:
            q = (1 - self.sparse_decay) * tf_hprob
        tf_prev_hprob = tf_hprob
        sparse_grad = self.sparse_level - q
        tf_hg = self.sparse_weight * tf.reduce_mean(sparse_grad, axis=0, keep_dims=True)
        tf_wg = self.sparse_weight * tf.matmul(tf.transpose(tf_x), sparse_grad) / tf.to_float(
            tf.shape(tf_x)[0])
        return tf_hg, tf_wg, tf_prev_hprob

    # def _get_positive_grad(self, x, hprob):
    def _create_positive_grad(self, tf_x, tf_hprob):
        tf_hg = tf.reduce_mean(tf_hprob, axis=0, keep_dims=True)
        tf_vg = tf.reduce_mean(tf_x, axis=0, keep_dims=True)
        tf_wg = tf.matmul(tf.transpose(tf_x), tf_hprob) / tf.to_float(tf.shape(tf_x)[0])
        return tf_hg, tf_vg, tf_wg

    def _create_negative_grad(self, tf_x, tf_hprob):
        return self._create_positive_grad(tf_x, tf_hprob)

    @abc.abstractmethod
    # def _get_hidden_prob(self, vsample, **kwargs):
    def _create_hidden_prob(self, vsample, **kwargs):
        pass

    @abc.abstractmethod
    # def _sample_hidden(self, hprob):
    def _create_sample_hidden(self, tf_hprob):
        pass

    @abc.abstractmethod
    # def _get_visible_prob(self, hsample):
    def _create_visible_prob(self):
        pass

    @abc.abstractmethod
    # def _sample_visible(self, vprob):
    def _create_sample_visible(self, tf_vprob):
        pass

    def _get_params_from_tf_params(self, **kwargs):
        sess = self._get_session(**kwargs)
        v, h, w = sess.run([self.tf_v_, self.tf_h_, self.tf_w_])
        # vgrad_inc, hgrad_inc, wgrad_inc = sess.run([self.tf_vgrad_inc_, self.tf_hgrad_inc_,
        #                                            self.tf_wgrad_inc_])

        if sess != self.tf_session_:
            sess.close()
        # return v, h, w, vgrad_inc, hgrad_inc, wgrad_inc
        return v, h, w

    def get_free_energy(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        fe = sess.run(self.tf_free_energy_, feed_dict={self.tf_x_: x})
        if sess != self.tf_session_:
            sess.close()
        return fe

    def _create_free_energy(self, tf_x):
        pass

    def get_logpartition(self, method='exact'):
        print('get_logpartition is not implemented\n')
        pass

    # def _create_generate_data(self, num_samples=100, num_gibbs_steps=1000,
    #                   num_burnin_steps=1000, num_intervals=1000, to_use_multichain=True):
    #     if to_use_multichain:
    #         # hprob = np.random.rand(num_samples, self.num_hidden)
    #         tf_hprob = 0.5 * tf.ones(shape=[num_samples, self.num_hidden], dtype=tf.float32)
    #
    #         for i in range(num_gibbs_steps + 1):
    #             print(i)
    #             tf_hsample, tf_vprob, tf_vsample, tf_hprob = self._create_gibbs_sampling(tf_hprob)
    #         return tf_vsample
    #     else:
    #         return None



    def get_reconstruction_error(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        err = sess.run(self.tf_reconstruction_error_, feed_dict={self.tf_x_: x})
        if sess != self.tf_session_:
            sess.close()
        return err

    def _create_reconstruction_error(self, tf_x, tf_rdata=None):
        return tf.abs(tf_x - self._create_reconstruction(tf_x)) if tf_rdata is None else tf.abs(
            tf_x - tf_rdata)

    def get_reconstruction_loglik(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        err = sess.run(self.tf_reconstruction_loglik_, feed_dict={self.tf_x_: x})
        if sess != self.tf_session_:
            sess.close()
        return err

    def _create_reconstruction_loglik(self, tf_x, rdata=None):
        pass

    def get_reconstruction(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        x_recon = sess.run(self.tf_reconstruction_, feed_dict={self.tf_x_: x})
        if sess != self.tf_session_:
            sess.close()
        return x_recon

    def _create_reconstruction(self, tf_x):
        # hprob = self._get_hidden_prob(x)
        # return self._get_visible_prob(hprob)
        tf_hprob = self._create_hidden_prob(tf_x)
        return self._create_visible_prob(tf_hprob)

    def _on_batch_end(self, x, y=None, **kwargs):
        outs = super(TensorFlowRBM, self)._on_batch_end(x, y=y)
        for m in self.metrics:
            if m == 'recon_err':
                outs.update({m: np.sum(self.get_reconstruction_error(
                    x, rdata=kwargs['rdata'] if 'rdata' in kwargs else None)) / x.size})
            if m == 'free_energy':
                outs.update({m: self.get_free_energy(x).mean()})
            if m == 'recon_loglik':
                outs.update({m: self.get_reconstruction_loglik(x).mean()})
            if m == 'loglik_csl':
                outs.update({m: self.get_loglik(x, method='csl').mean()})
        return outs

    def _create_learning_rate(self, tf_epoch):
        # adjust learning rate
        if self.learning_rate_decay == DECAY_METHOD['linear']:
            tf_learning_rate_update = self.tf_learning_rate_.assign(self.learning_rate0_
                                                                    - self.learning_rate_decay_rate * self.learning_rate0_ / self.num_epochs)
        elif self.learning_rate_decay == DECAY_METHOD['div_sqrt']:
            # tf_learning_rate = self.learning_rate0_ / np.sqrt(tf_epoch)
            tf_learning_rate_update = tf.truediv(self.learning_rate0_, tf.sqrt(tf_epoch))
        elif self.learning_rate_decay == DECAY_METHOD['exp']:
            tf_learning_rate_update = self.tf_learning_rate_.assign(
                tf.mul(self.tf_learning_rate_, self.learning_rate_decay_rate))
        else:
            tf_learning_rate_update = None
        return tf_learning_rate_update

    def _create_momentum(self, tf_epoch):
        if self.momentum_method == MOMENTUM_METHOD['sudden']:
            tf_momentum_update = tf.cond(tf.greater_equal(tf_epoch, self.momentum_iteration),
                                         lambda: self.tf_momentum_.assign(self.final_momentum),
                                         lambda: self.tf_momentum_)
        else:
            tf_momentum_update = None
        return tf_momentum_update

    def _on_epoch_end(self):
        super(TensorFlowRBM, self)._on_epoch_end()
        if self.tf_learning_rate_update_ is not None:
            self.tf_session_.run(self.tf_learning_rate_update_,
                                 feed_dict={self.tf_epoch_: self.epoch_})

        if self.tf_momentum_update_ is not None:
            self.tf_session_.run(self.tf_momentum_update_,
                                 feed_dict={self.tf_epoch_: self.epoch_})
        lr, mo = self.tf_session_.run([self.tf_learning_rate_, self.tf_momentum_])
        print('learning rate = %f momentum=%f\n' % (lr, mo))

    def disp_filters(self, num_filters=100, filter_idx=None, disp_dim=None,
                     tile_shape=(10, 10), output_pixel_vals=False, **kwargs):
        if disp_dim is None:
            n = int(np.sqrt(self.num_visible))
            disp_dim = (n, n)
        else:
            assert len(disp_dim) == 2
        n = np.prod(disp_dim)

        assert num_filters == np.prod(tile_shape)

        if filter_idx is None:
            filter_idx = np.random.permutation(self.num_hidden)[:num_filters]

        sess = self._get_session(**kwargs)
        w = sess.run(self.tf_w_).T[filter_idx, :n]
        if sess != self.tf_session_:
            sess.close()

        img = tile_raster_images(w, img_shape=disp_dim, tile_shape=tile_shape, tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=output_pixel_vals)

        if 'ax' in kwargs:
            ax = kwargs['ax']
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            ax.grid(0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("epoch #{}".format(kwargs['epoch']), fontsize=28)
        else:
            fig, ax = plt.subplots()
            ax.set_title(kwargs['title'] if 'title' in kwargs else "Receptive fields",
                         fontsize=28)
            ax.axis('off')
            plt.colorbar()
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            plt.show()


    def disp_recondata(self, data=None, num_images=50, filter_idx=None, disp_dim=None,
                     tile_shape=(10, 10), output_pixel_vals=False, **kwargs):
        if disp_dim is None:
            n = int(np.sqrt(self.num_visible))
            disp_dim = (n, n)
        else:
            assert len(disp_dim) == 2
        n = np.prod(disp_dim)

        assert num_images*2 == np.prod(tile_shape)

        if filter_idx is None:
            filter_idx = np.random.permutation(self.num_hidden)[:num_images]

        rdata = self.get_reconstruction(data)
        img_disp = np.zeros([2*num_images, self.num_visible])
        for i in range(num_images):
            img_disp[2*i,:] = data[i,:]
            img_disp[2*i + 1, :] = rdata[i, :]

        img = tile_raster_images(img_disp, img_shape=disp_dim, tile_shape=tile_shape, tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=output_pixel_vals)

        if 'ax' in kwargs:
            ax = kwargs['ax']
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            ax.grid(0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("epoch #{}".format(kwargs['epoch']), fontsize=28)
        else:
            fig, ax = plt.subplots()
            ax.set_title(kwargs['title'] if 'title' in kwargs else "Receptive fields",
                         fontsize=28)
            ax.axis('off')
            plt.colorbar()
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            plt.show()

    def display(self, param, **kwargs):
        if param == 'filters':
            self.disp_filters(**kwargs)
        elif param == 'reconstruction':
            self.disp_recondata(**kwargs)
        else:
            print(param)
            raise NotImplementedError

    def get_params(self, deep=True):
        out = super(TensorFlowRBM, self).get_params(deep=deep)
        out.update({'num_hidden': self.num_hidden,
                    'num_visible': self.num_visible,
                    'learning_method': self.learning_method,
                    'num_cd': self.num_cd,
                    'sampling_in_last_cd': self.sampling_in_last_cd,
                    'num_pcd': self.num_pcd,
                    'learning_rate': self.learning_rate,
                    'learning_rate_decay': self.learning_rate_decay,
                    'learning_rate_decay_rate': self.learning_rate_decay_rate,
                    'h_init': self.h_init, 'v_init': self.v_init, 'w_init': self.w_init,
                    'momentum_method': self.momentum_method,
                    'initial_momentum': self.initial_momentum,
                    'final_momentum': self.final_momentum,
                    'momentum_iteration': self.momentum_iteration,
                    'weight_cost': self.weight_cost,
                    'sparse_weight': self.sparse_weight,
                    'sparse_level': self.sparse_level,
                    'sparse_decay': self.sparse_decay})
        return out

    def get_all_params(self, deep=True):
        out = super(TensorFlowRBM, self).get_all_params(deep=deep)
        out.update(self.get_params(deep=deep))
        out.update({'learning_rate0_': self.learning_rate0_,
                    'momentum_': self.momentum_,
                    'h_': copy.deepcopy(self.h_),
                    'v_': copy.deepcopy(self.v_),
                    'w_': copy.deepcopy(self.w_),
                    'hgrad_inc_': copy.deepcopy(self.hgrad_inc_),
                    'vgrad_inc_': copy.deepcopy(self.vgrad_inc_),
                    'wgrad_inc_': copy.deepcopy(self.wgrad_inc_)})
        return out
