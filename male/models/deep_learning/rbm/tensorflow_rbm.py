from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import abc

import numpy as np
import tensorflow as tf

from sklearn.utils.validation import check_is_fitted

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


class TensorFlowRBM(TensorFlowModel):
    """A generic class of Restricted Boltzmann Machine using TensorFlow
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model_name="TensorFlowRBM",
                 num_hidden=100,
                 num_visible=500,
                 # learning method
                 learning_method='cd',
                 num_cd=1,
                 sampling_in_last_cd='none',
                 num_pcd=15,
                 # learning rates
                 learning_rate=0.1,
                 learning_rate_decay='none',
                 learning_rate_decay_rate=0.0,
                 # parameter initialization
                 h_init=0.0, v_init=0.0, w_init=0.1,
                 # momentum
                 momentum_method='none',
                 initial_momentum=0.5,
                 final_momentum=0.9,
                 momentum_iteration=5,
                 # regularization
                 weight_cost=0.0,
                 sparse_weight=0.0,
                 sparse_level=0.1,
                 sparse_decay=0.9,
                 **kwargs):

        super(TensorFlowRBM, self).__init__(model_name=model_name, **kwargs)

        self.num_hidden = num_hidden
        self.num_visible = num_visible

        self.learning_method = learning_method
        self.num_cd = num_cd
        self.sampling_in_last_cd = sampling_in_last_cd
        self.num_pcd = num_pcd

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_rate = learning_rate_decay_rate

        self.h_init, self.v_init, self.w_init = h_init, v_init, w_init

        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.momentum_iteration = momentum_iteration
        self.momentum_method = momentum_method

        self.weight_cost = weight_cost
        self.sparse_weight = sparse_weight
        self.sparse_level = sparse_level
        self.sparse_decay = sparse_decay

    def _init(self):
        super(TensorFlowRBM, self)._init()

        try:
            self.learning_method = LEARNING_METHOD[self.learning_method]
        except KeyError:
            raise ValueError("Learning method %s is not supported." % self.learning_method)
        try:
            self.sampling_in_last_cd = CD_SAMPLING[self.sampling_in_last_cd]
        except KeyError:
            raise ValueError("CD sampling %s is not supported." % self.sampling_in_last_cd)
        try:
            self.learning_rate_decay = DECAY_METHOD[self.learning_rate_decay]
        except KeyError:
            raise ValueError("Learning rate decay method %s is not supported."
                             % self.learning_rate_decay)
        try:
            self.momentum_method = MOMENTUM_METHOD[self.momentum_method]
        except KeyError:
            raise ValueError("Momentum method %s is not supported." % self.momentum_method)

        self.learning_rate0 = self.learning_rate
        with self.tf_graph.as_default():
            self.tf_learning_rate = tf.get_variable(
                "learning_rate", shape=[1],
                initializer=tf.constant_initializer(self.learning_rate))
            self.momentum = tf.get_variable(
                "momentum", shape=[1],
                initializer=tf.constant_initializer(self.initial_momentum))

    def _init_params(self, x):
        # initialize parameters
        k, n = self.num_hidden, self.num_visible
        self.h = tf.get_variable("hidden_bias", shape=[1, k],
                                 initializer=tf.random_normal_initializer(stddev=self.h_init))
        self.v = tf.get_variable("visible_bias", shape=[1, n],
                                 initializer=tf.random_normal_initializer(stddev=self.v_init))
        self.w = tf.get_variable("weight", shape=[n, k],
                                 initializer=tf.random_normal_initializer(stddev=self.w_init))
        self.hgrad_inc = tf.get_variable("hidden_grad_inc", shape=[1, k],
                                         initializer=tf.constant_initializer(0.0))
        self.vgrad_inc = tf.get_variable("visible_grad_inc", shape=[1, n],
                                         initializer=tf.constant_initializer(0.0))
        self.wgrad_inc = tf.get_variable("weight_grad_inc", shape=[n, k],
                                         initializer=tf.constant_initializer(0.0))

    def _build_model(self, x):
        self.x = tf.placeholder(tf.float32, shape=[None, self.num_visible], name="data")

        self.hidden_prob = self._get_hidden_prob(self.x)
        self.reconstruction = self._create_reconstruction(self.x)
        self.reconstruction_error = self._create_reconstruction_error(self.x,
                                                                      rdata=self.reconstruction)
        self.free_energy = self._create_free_energy(self.x)

        self.reconstruction_loglik = self._create_reconstruction_loglik(self.x,
                                                                        rdata=self.reconstruction)

        pos_hgrad, pos_vgrad, pos_wgrad = self._initialize_grad()

        prev_hprob = tf.get_variable("previous_hidden_prob",
                                     shape=[self.batch_size, self.num_hidden],
                                     initializer=tf.constant_initializer(0.0))

        # ======= clamp phase ========
        hprob = self._get_hidden_prob(self.x)

        # sparsity
        if self.sparse_weight > 0:
            hg, wg, prev_hprob = self._hidden_sparsity(self.x, prev_hprob, hprob)
            pos_hgrad += hg
            pos_wgrad += wg

        hg, vg, wg = self._get_positive_grad(self.x, hprob)
        pos_hgrad += hg
        pos_vgrad += vg
        pos_wgrad += wg

        # ======== free phase =========
        if self.learning_method == LEARNING_METHOD['cd']:
            for icd in range(self.num_cd - 1):
                hsample, vprob, vsample, hprob = self._gibbs_sampling(
                    hprob, sampling=CD_SAMPLING['hidden_visible'])
            hsample, vprob, vsample, hprob = self._gibbs_sampling(
                hprob, sampling=self.sampling_in_last_cd)

        # ======== negative phase =========
        neg_hgrad, neg_vgrad, neg_wgrad = self._get_negative_grad(vprob, hprob)

        self.hgrad_inc = self.momentum * self.hgrad_inc + self.tf_learning_rate * (
            pos_hgrad - neg_hgrad)
        self.vgrad_inc = self.momentum * self.vgrad_inc + self.tf_learning_rate * (
            pos_vgrad - neg_vgrad)
        self.wgrad_inc = self.momentum * self.wgrad_inc \
                         + self.tf_learning_rate * (
            pos_wgrad - neg_wgrad - self.weight_cost * self.w)

        # update params
        self.h_update = self.h.assign_add(self.hgrad_inc)
        self.v_update = self.v.assign_add(self.vgrad_inc)
        self.w_update = self.w.assign_add(self.wgrad_inc)

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
        check_is_fitted(self, "w")
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
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]

                self.tf_session.run(
                    [self.w_update, self.h_update, self.v_update],
                    feed_dict={self.x: x_batch})

                batch_logs.update(self._on_batch_end(x_batch))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid)
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _gibbs_sampling(self, hprob, sampling=CD_SAMPLING['hidden_visible']):
        if sampling == CD_SAMPLING['hidden']:
            hsample = self._sample_hidden(hprob)
            vprob = self._get_visible_prob(hsample)
            vsample = tf.identity(vprob)  # copy
            hprob = self._get_hidden_prob(vsample)
        elif sampling == CD_SAMPLING['none']:
            hsample = tf.identity(hprob)  # copy
            vprob = self._get_visible_prob(hsample)
            vsample = tf.identity(vprob)  # copy
            hprob = self._get_hidden_prob(vsample)
        elif sampling == CD_SAMPLING['hidden_visible']:
            hsample = self._sample_hidden(hprob)
            vprob = self._get_visible_prob(hsample)
            vsample = self._sample_visible(vprob)
            hprob = self._get_hidden_prob(vsample)
        else:
            hsample = tf.identity(hprob)  # copy
            vprob = self._get_visible_prob(hsample)
            vsample = self._sample_visible(vprob)
            hprob = self._get_hidden_prob(vsample)

        return hsample, vprob, vsample, hprob

    def _initialize_grad(self):
        k, n = self.num_hidden, self.num_visible
        pos_hgrad = tf.get_variable("positive_hidden_grad", shape=[1, k],
                                    initializer=tf.constant_initializer(0.0))
        pos_vgrad = tf.get_variable("positive_visible_grad", shape=[1, n],
                                    initializer=tf.constant_initializer(0.0))
        pos_wgrad = tf.get_variable("positive_weight_grad", shape=[n, k],
                                    initializer=tf.constant_initializer(0.0))
        return pos_hgrad, pos_vgrad, pos_wgrad

    def _hidden_sparsity(self, x, prev_hprob, hprob):
        if prev_hprob.get_shape() == hprob.get_shape():
            q = self.sparse_decay * prev_hprob + (1 - self.sparse_decay) * hprob
        else:
            q = (1 - self.sparse_decay) * hprob
        prev_hprob = hprob
        sparse_grad = self.sparse_level - q
        hg = self.sparse_weight * tf.reduce_mean(sparse_grad, axis=0, keep_dims=True)
        wg = self.sparse_weight * tf.matmul(tf.transpose(x), sparse_grad) / tf.to_float(
            tf.shape(x)[0])
        return hg, wg, prev_hprob

    def _get_positive_grad(self, x, hprob):
        hg = tf.reduce_mean(hprob, axis=0, keep_dims=True)
        vg = tf.reduce_mean(x, axis=0, keep_dims=True)
        wg = tf.matmul(tf.transpose(x), hprob) / tf.to_float(tf.shape(x)[0])
        return hg, vg, wg

    def _get_negative_grad(self, x, hprob):
        return self._get_positive_grad(x, hprob)

    @abc.abstractmethod
    def _get_hidden_prob(self, vsample, **kwargs):
        pass

    @abc.abstractmethod
    def _sample_hidden(self, hprob):
        pass

    @abc.abstractmethod
    def _get_visible_prob(self, hsample):
        pass

    @abc.abstractmethod
    def _sample_visible(self, vprob):
        pass

    def get_free_energy(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        fe = sess.run(self.free_energy, feed_dict={self.x: x})
        if sess != self.tf_session:
            sess.close()
        return fe

    def _create_free_energy(self, x):
        pass

    def get_csl(self, x, num_hidden_samples=1000, num_steps=100):
        hprob = 0.5 * np.ones([num_hidden_samples, self.num_hidden])
        hsample = self._sample_hidden(hprob)
        for i in range(num_steps):
            hsample, vprob, vsample, hprob = self._gibbs_sampling(hprob)
        return self.get_conditional_loglik(x, hsample)

    def get_conditional_loglik(self, x, hsample):
        pass

    def get_loglik(self, x, method='csl', **kwargs):
        if method == 'csl':
            return self.get_csl(x, **kwargs)
        elif method == 'exact':
            logZ = self.get_logpartition()
            return (-self.get_free_energy(x)) - logZ
        else:
            raise NotImplementedError

    def get_logpartition(self, method='exact'):
        pass

    def generate_data(self, num_samples=100, num_gibbs_steps=1000,
                      num_burnin_steps=1000, num_intervals=1000, to_use_multichain=True):
        if to_use_multichain:
            # hprob = np.random.rand(num_samples, self.num_hidden)
            hprob = 0.5 * np.ones((num_samples, self.num_hidden))
            for i in range(num_gibbs_steps + 1):
                print(i)
                hsample, vprob, vsample, hprob = self._gibbs_sampling(hprob)
            return vsample
        else:
            return None

    def get_reconstruction_error(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        err = sess.run(self.reconstruction_error, feed_dict={self.x: x})
        if sess != self.tf_session:
            sess.close()
        return err

    def _create_reconstruction_error(self, x, rdata=None):
        return tf.abs(x - self._create_reconstruction(x)) if rdata is None else tf.abs(x - rdata)

    def get_reconstruction_loglik(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        err = sess.run(self.reconstruction_loglik, feed_dict={self.x: x})
        if sess != self.tf_session:
            sess.close()
        return err

    def _create_reconstruction_loglik(self, x, rdata=None):
        pass

    def get_reconstruction(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        x_recon = sess.run(self.reconstruction, feed_dict={self.x: x})
        if sess != self.tf_session:
            sess.close()
        return x_recon

    def _create_reconstruction(self, x):
        hprob = self._get_hidden_prob(x)
        return self._get_visible_prob(hprob)

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

    def _on_epoch_end(self):
        super(TensorFlowRBM, self)._on_epoch_end()

        # adjust learning rate
        if self.learning_rate_decay == DECAY_METHOD['linear']:
            self.learning_rate = (self.learning_rate0
                                  - self.learning_rate_decay_rate * self.learning_rate0
                                  / self.num_epochs)
        elif self.learning_rate_decay == DECAY_METHOD['div_sqrt']:
            self.learning_rate = self.learning_rate0 / np.sqrt(self.epoch)
        elif self.learning_rate_decay == DECAY_METHOD['exp']:
            self.learning_rate *= self.learning_rate_decay_rate

        # adjust momentum
        if self.momentum_method == MOMENTUM_METHOD['sudden']:
            if self.epoch >= self.momentum_iteration:
                self.momentum = self.final_momentum

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
        w = sess.run(self.w).T[filter_idx, :n]
        if sess != self.tf_session:
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

    def display(self, param, **kwargs):
        if param == 'filters':
            self.disp_filters(**kwargs)
        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        out = super(TensorFlowRBM, self).get_params(deep=deep)
        param_names = TensorFlowRBM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
