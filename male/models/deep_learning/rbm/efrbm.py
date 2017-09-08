from __future__ import division, print_function, absolute_import
import numpy as np
from matplotlib import pyplot as plt
from ....utils.generic_utils import make_batches
from ....utils.disp_utils import tile_raster_images
from .rbm import RBM, LEARNING_METHOD, CD_SAMPLING


#################################################################################
#                       EFLayer <------------------------- EFRBM                #
#                      /      \                                                 #
#                    /         \                                                #
#    CategoricalLayer           ContinuousLayer                                 #
#            /                                                                  #
#          /                                                                    #
#       BinaryLayer                                                             #
#################################################################################
class EFLayer():
    def __init__(self,
                 name='EFLayer',
                 suf_stat_dim=1,
                 **kwargs):
        self.name = name
        super(EFLayer, self).__init__(**kwargs)
        self.suf_stat_dim = suf_stat_dim
        self.hidden_sparsity = False

    def get_additional_func(self, x):
        pass

    def get_sufficient_statistics_func(self, x):
        pass

    def get_partition_func(self, params):
        pass

    def sample_sufficient_stat(self, params_cond):
        pass

    def get_cond_dist_params(self, layer_input, bias=None):
        # Input:
        #   layer_input :               #samples x #units x #sufficient statistics dim
        #   bias        :               1 x #units x #sufficient statistics dim
        # Output: distribution parameters of conditional distribution
        #                               #samples x #units x #sufficient statistics dim
        pass

    def hid_suf_stat_posterior_expectation(self, params_cond):
        pass

    def get_weight_coff(self, bias=None):
        return np.ones(bias.shape)

    def get_visible_energy_deri_expectation(self, sufficient_stat_visible, bias=None,
                                            other_layer_input=None):
        # Input:    sufficient_stat_visible     # #samples x # uints x # sufficient statistics dim
        return sufficient_stat_visible

    def get_hidden_energy_deri_expectation(self, sufficient_stat_visible, params_cond,
                                           bias=None, other_layer_input=None):
        return self.hid_suf_stat_posterior_expectation(params_cond)

    def init_params(self, bias, weight_matrix, layer):  # layer=0: visible, =1: hidden
        return bias, weight_matrix

    def sufficient_stat_to_value(self, sufficient_stat):
        pass

    def get_individual_layer_prob(self, x, params):  # q(bx;\btheta)
        # Compute q(bx;\btheta) = exp[\sum_i [\log r_i (x_i)
        #                             +sum_j \theta_i,j *f_i^j (x) - A_i (\theta_i) ] ]
        # Input
        #   x       : input data                            #samples x #units
        #   params  : original distribution parameters or conditional distribution parameters
        #                                                   # #samples x #units x #suf
        # Output
        #   p       : probability q(bx;\btheta)             # samples x #units

        num_data = x.shape[0]
        feat = self.get_sufficient_statistics_func(x)  # #samples x #units x # suf

        a = np.multiply(params, feat)  # #samples x #units x #suf
        b = np.sum(a, axis=2, keepdims=False)  # #samples x #units

        add_feat = self.get_additional_func(x)  # 1 x #units
        partition = self.get_partition_func(params)  # vector:# (1|num_data) x # units

        p = np.exp(np.log(add_feat) + b - partition)  # #samples x # units
        return p

    def get_individual_params(self, bias):
        # the parameters of exponential family distribution
        # when the layer isn't connected with other layers
        # for binary and categorical: exponential family params (theta) = model params = bias
        # for continuous: params differ from bias
        return bias

    def get_hidden_activation(self, params_cond):
        pass


class CategoricalLayer(EFLayer):
    def __init__(self,
                 name='CategoricalLayer',
                 **kwargs):
        super(CategoricalLayer, self).__init__(name=name, **kwargs)
        self.hidden_sparsity = True

    def get_additional_func(self, x):
        return 1.0

    def get_sufficient_statistics_func(self, x):
        assert len(x.shape) == 2
        num_data, num_units = x.shape

        # ensure that x values should be in [0, L-1]
        assert np.amax(x) < self.suf_stat_dim and np.amin(x) >= 0
        feat = np.zeros([num_data, num_units, self.suf_stat_dim])

        for i in range(num_data):
            for j in range(num_units):
                idx = int(round(x[i, j]))
                feat[i, j, idx] = 1.0
        return feat

    def get_partition_func(self, params):  # 1 x #units x #sufficient statistics dim
        e = np.exp(params)
        return np.log(np.sum(e, axis=2, keepdims=False))

    def get_cond_dist_params(self, layer_input, bias=None):
        # Input:
        #   layer_input                 #num_data x #units x #sufficient statistics dim
        #   params: the bias matrix     #num_data x #units x #sufficient statistics dim
        # Output: distribution parameters of conditional distribution
        #                      #num_data x #units x #sufficient statistics dim

        theta_cond = bias + layer_input

        e = np.exp(theta_cond)  # #samples x #units x #sufficient statistics dim
        s = np.sum(e, axis=2, keepdims=True)  # #samples x #units
        s_tile = np.tile(s, [1, 1, e.shape[2]])
        return e / s_tile

    def hid_suf_stat_posterior_expectation(self, params_cond):
        # Input:
        #   params_cond             #samples x #units x #sufficient statistics dim
        # Output:   mean of multinomial distribution which is exact parameters of the distribution
        #                           #samples x #units x #sufficient statistics dim
        return params_cond

    def sample_sufficient_stat(self, params_cond):
        # Input: params_cond            # #samples x #units x #sufficient statistics dim
        # Output: samples               # #samples x #units x #sufficient statistics dim
        mean_cond = params_cond
        num_samples, num_units, _ = params_cond.shape
        samples = np.zeros(params_cond.shape)
        for i in range(num_samples):
            for j in range(num_units):
                counts = np.random.multinomial(1, mean_cond[i, j, :])
                samples[i, j, :] = counts
        return samples

    def sufficient_stat_to_value(self, sufficient_stat):
        # Input:    sufficient_stat        #sample x #units x #sufficient state dim
        # Output:   samples               #sample x #units
        samples = np.argmax(sufficient_stat, axis=2)
        return samples

    def get_hidden_activation(self, params_cond):
        return params_cond


class BinaryLayer(CategoricalLayer):
    def __init__(self,
                 name='BinaryLayer',
                 **kwargs):
        super(BinaryLayer, self).__init__(name=name, **kwargs)
        self.suf_stat_dim = 1
        self.hidden_sparsity = True

    def get_sufficient_statistics_func(self, x):
        assert len(x.shape) == 2
        feat = x.copy()
        feat = np.reshape(feat, [*x.shape, 1])
        return feat

    def get_partition_func(self, params):
        s = np.log(1 + np.exp(params))
        return np.sum(s, axis=2, keepdims=False)

    def get_cond_dist_params(self, layer_input, bias=None):
        # Input:
        #   layer_input :                   #num_data x #units x #sufficient statistics dim
        #   params      :the bias matrix    #num_data x #units x #sufficient statistics dim
        # Output:                           #num_data x #units x #sufficient statistics dim
        theta_cond = bias + layer_input
        return 1.0 / (1.0 + np.exp(-theta_cond))

    def sample_sufficient_stat(self, params_cond):
        # Input: params_cond            # #samples x #units x 1
        # Output: samples               # #samples x #units x 1
        mean_cond = params_cond  # #samples x #units x 1
        samples = (mean_cond > np.random.rand(*mean_cond.shape)).astype(np.int)
        return samples

    def sufficient_stat_to_value(self, sufficient_stat):
        # Input: sufficient_stat        #sample x #units x #sufficient state dim
        # Output: samples               #sample x #units
        num_data, num_units, num_sufficient_stat = sufficient_stat.shape
        samples = np.reshape(sufficient_stat, [num_data, num_units])
        return samples


class ContinuousLayer(EFLayer):
    def __init__(self,
                 name='ContinuousLayer',
                 trainable_sigma2=True,
                 **kwargs):
        super(ContinuousLayer, self).__init__(name=name, **kwargs)
        self.suf_stat_dim = 2
        self.trainable_sigma2 = trainable_sigma2

    def get_additional_func(self, x):
        return 1.0 / np.sqrt(2 * np.pi)

    def get_sufficient_statistics_func(self, x):
        # Input:  x                                  # samples x # units
        # Output: sufficient statistics of x        # samples x # units x 2
        x2 = np.square(x)
        return np.stack((x2, x), axis=2)  # samples x # units x 2

    def get_partition_func(self, params):
        # Input: params                             # 1 x # units x 2
        # Output: partition function                # 1 x # units
        param1 = params[:, :, 0]
        param2 = params[:, :, 1]
        partition = -np.true_divide(np.square(param2), 4.0 * param1) - 0.5 * np.log(
            -2.0 * param1)
        return partition  # 1 x # units

    def get_cond_dist_params(self, layer_input, bias=None):
        # get true parameters of conditional distributions
        # Input:    layer_input of size         # num_data x #units x #sufficient statistics dim
        #           params: the bias matrix     # 1 x units x 2
        # Output: [mu, sigma2]            #samples x #units x 2
        num_data = layer_input.shape[0]

        bias1 = bias[:, :, 0]  # 1 x # units
        bias1_exp = np.exp(bias1)  # 1 x # units

        bias2 = bias[:, :, 1]  # 1 x # units

        sigma2_cond = np.tile(bias1_exp, [num_data, 1])  # #samples x #units
        mean_cond = (np.tile(bias2, [num_data, 1]) +
                     np.multiply(np.tile(np.sqrt(bias1_exp), [num_data, 1]),
                                 layer_input[:, :, 1]))

        return np.stack((mean_cond, sigma2_cond), axis=2)  # #samples x #units x 2

    def hid_suf_stat_posterior_expectation(self, params_cond):
        # Input: params_cond                                #samples x #units x 2
        # Output: mean and variance of conditional dist     #samples x #units x 2
        mean_cond = params_cond[:, :, 0]
        sigma2_cond = params_cond[:, :, 1]
        m = np.zeros(params_cond.shape)  # #samples x #units x 2
        m[:, :, 0] = mean_cond + sigma2_cond
        m[:, :, 1] = mean_cond
        return m

    def get_weight_coff(self, bias=None):
        # Input: bias            # 1 x # units x 2
        # Output: coff           # 1 x # units x 1
        #   coff is the coefficient of Gaussian layer that is 1/sqrt(2 x pi x exp(bias1))
        bias1 = bias[:, :, 0]
        bias1_exp = np.exp(bias1)
        coff = np.zeros(bias.shape)
        coff[:, :, 1] = np.true_divide(1.0, np.sqrt(bias1_exp))
        return coff

    def get_visible_energy_deri_expectation(self, sufficient_stat_visible, bias=None,
                                            other_layer_input=None):
        # Input:    sufficient_stat_visible     # #samples x # this layer uints x # sufficient statistics dim
        # Input:    other_layer_input           # #samples x  # this layer uints x # sufficiient statistics dim
        # Output:   E_p(h|v;\bpsi)[-\partial E /\partial \theta_i]
        bias1 = bias[:, :, 0]  # #samples x #units x 1
        bias1_exp = np.exp(bias1)  # #samples x #units x 1
        bias2 = bias[:, :, 1]  # #samples x #units x 1

        # feat2 = sufficient_stat_visible[:, :, 0]      # #samples x #units x 1
        feat = sufficient_stat_visible[:, :, 1]  # #samples x #units x 1
        feat2 = np.square(feat)

        # #samples x #units x # sufficient statistics dim
        res = np.zeros(sufficient_stat_visible.shape)

        # t1 = 0.5 * np.true_divide(feat2 - 2.0 * np.multiply(bias2, feat),
        #                      np.square(bias1_exp))
        #
        # t2 = 0.5 * np.true_divide( np.multiply(feat, other_layer_input[:, :, 1]),
        #                            np.power( np.sqrt(bias1_exp), 3) )
        #
        # if self.trainable_sigma2 == True:
        #     res[:, :, 0] = np.multiply(t1 - t2, bias1_exp)

        t1 = 0.5 * np.true_divide(feat2 - 2.0 * np.multiply(bias2, feat), bias1_exp)

        t2 = 0.5 * np.true_divide(np.multiply(feat, other_layer_input[:, :, 1]), np.sqrt(bias1_exp))

        if self.trainable_sigma2 == True:
            res[:, :, 0] = t1 - t2

        res[:, :, 1] = np.true_divide(feat, bias1_exp)
        return res

    def get_hidden_energy_deri_expectation(self, sufficient_stat_visible, params_cond,
                                           bias=None, other_layer_input=None):
        # Input:
        #   sufficient_stat_visible:            # #samples x #visibile units x # sufficient statistics dim
        #   params_cond:                        # #samples x #hidden units x 2
        #   bias:                               # 1 x #units x 2
        # Output:                               #samples x # units x # 2

        # #samples x #hidden units x 2
        expectation = self.hid_suf_stat_posterior_expectation(params_cond)
        # expectation_feat2 = expectation[:, :, 0]      # #samples x #hidden units x 1
        expectation_feat = expectation[:, :, 1]  # #samples x #hidden units x 1
        expectation_feat2 = np.square(expectation_feat)

        # 1 x #hidden units x 1
        bias1 = bias[:, :, 0]
        bias1_exp = np.exp(bias1)
        # 1 x #hidden units x 1
        bias2 = bias[:, :, 1]

        res = np.zeros(params_cond.shape)

        # t1 =0.5 * np.true_divide(
        #     expectation_feat2 - 2.0 * np.multiply(bias2, expectation_feat),
        #     np.square(bias1_exp))
        #
        # t2 = 0.5 * np.true_divide( np.multiply(expectation_feat, other_layer_input[:, :, 1]),
        #                            np.power(np.sqrt(bias1_exp) , 3))
        # if self.trainable_sigma2 == True:
        #     res[:, :, 0] = np.multiply(t1 - t2, bias1_exp)
        t1 = 0.5 * np.true_divide(
            expectation_feat2 - 2.0 * np.multiply(bias2, expectation_feat), bias1_exp)

        t2 = 0.5 * np.true_divide(
            np.multiply(expectation_feat, other_layer_input[:, :, 1]),
            np.sqrt(bias1_exp))

        if self.trainable_sigma2 == True:
            res[:, :, 0] = t1 - t2

        res[:, :, 1] = np.true_divide(expectation_feat, bias1_exp)
        return res

    def sample_sufficient_stat(self, params_cond):
        # Input: params_cond                    # #samples x #units x 2
        # Output: samples                       # #samples x #units x 2

        mean_cond = params_cond[:, :, 0]  # #samples x #units x 1
        sigma2_cond = params_cond[:, :, 1]  # #samples x #units x 1

        mean_cond_sqz = np.squeeze(mean_cond)
        sigma2_cond_sqz = np.squeeze(sigma2_cond)

        num_samples, num_units, _ = params_cond.shape
        samples = np.zeros(params_cond.shape)
        for i in range(num_samples):
            s = np.random.multivariate_normal(mean_cond_sqz[i, :], np.diag(sigma2_cond_sqz[i, :]),
                                              size=[1])
            samples[i, :, 0] = np.square(s)
            samples[i, :, 1] = s
        return samples

    def init_params(self, bias, weight_matrix, layer):
        # Input:
        #   bias and weight_matrix: the bias of the layer and weight matrix connected to it
        #   layer=0: the layer is visible layer or =1 hidden layer
        if layer == 0:
            weight_matrix[:, :, 0, :] = 0.0
        if layer == 1:
            weight_matrix[:, :, :, 0] = 0.0
        return bias, weight_matrix

    def sufficient_stat_to_value(self, sufficient_stat):
        # Input: sufficient_stat           # #samples x #units x #sufficient statistics dims
        return sufficient_stat[:, :, 1]

    def get_individual_params(self, bias):
        # Input: params                             # 1 x #units x 2
        indi_params = np.zeros(bias.shape)
        bias1 = bias[:, :, 0]
        bias1_exp = np.exp(bias1)
        bias2 = bias[:, :, 1]
        indi_params[:, :, 0] = -0.5 / bias1_exp
        indi_params[:, :, 1] = bias2 / bias1_exp
        return indi_params


class EFRBM(RBM):
    def __init__(self,
                 model_name='EFRBM',
                 # dimentions of visible sufficient statistics function
                 suf_stat_dim_vis=1,
                 # dimentions of hidden sufficient statistics function
                 suf_stat_dim_hid=1,
                 visible_layer_type='binary',
                 hidden_layer_type='categorical',
                 Gaussian_layer_trainable_sigmal2=True,
                 **kwargs):
        super(EFRBM, self).__init__(model_name=model_name, **kwargs)
        self.suf_stat_dim_vis = suf_stat_dim_vis
        self.suf_stat_dim_hid = suf_stat_dim_hid
        self.visible_layer_type = visible_layer_type
        self.hidden_layer_type = hidden_layer_type
        self.Gaussian_layer_trainable_sigmal2 = Gaussian_layer_trainable_sigmal2

    def _init(self):
        super(EFRBM, self)._init()

        self.visible_layer = self._create_layer(self.visible_layer_type,
                                                suf_stat_dim=self.suf_stat_dim_vis)
        self.suf_stat_dim_vis = self.visible_layer.suf_stat_dim
        self.hidden_layer = self._create_layer(self.hidden_layer_type,
                                               suf_stat_dim=self.suf_stat_dim_hid)
        self.suf_stat_dim_hid = self.hidden_layer.suf_stat_dim

    def _create_layer(self, layer_type, suf_stat_dim):
        if layer_type == 'binary':
            return BinaryLayer(suf_stat_dim=1)
        elif layer_type == 'categorical':
            return CategoricalLayer(suf_stat_dim=suf_stat_dim)
        elif layer_type == 'continuous':
            return ContinuousLayer(suf_stat_dim=2,
                                   trainable_sigma2=self.Gaussian_layer_trainable_sigmal2)
        return None

    def _init_params(self, x):
        self.h = self.h_init * np.random.randn(1, self.num_hidden, self.suf_stat_dim_hid)

        self.v = self.v_init * np.random.randn(1, self.num_visible,
                                               self.suf_stat_dim_vis)

        self.w = self.w_init * np.random.randn(self.num_visible,
                                               self.num_hidden,
                                               self.suf_stat_dim_vis,
                                               self.suf_stat_dim_hid)

        self.hgrad_inc = np.zeros([1, self.num_hidden, self.suf_stat_dim_hid])
        self.vgrad_inc = np.zeros([1, self.num_visible, self.suf_stat_dim_vis])
        self.wgrad_inc = np.zeros([self.num_visible,
                                   self.num_hidden,
                                   self.suf_stat_dim_vis,
                                   self.suf_stat_dim_hid])

        self.v, self.w = self.visible_layer.init_params(self.v, self.w, layer=0)
        self.h, self.w = self.hidden_layer.init_params(self.h, self.w, layer=1)

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
        num_data = x.shape[0]
        k, n = self.num_hidden, self.num_visible
        prev_hid_act = np.zeros([1, k])

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]

                pos_hgrad, pos_vgrad, pos_wgrad = self._initialize_grad()

                # ======= positive phase ========
                sufficient_stat_batch = self.visible_layer.get_sufficient_statistics_func(
                    x_batch)
                hidden_input = self._forward(sufficient_stat_batch)

                params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input, self.h)
                hid_suf_stat_mean = self.hidden_layer.hid_suf_stat_posterior_expectation(
                    params_hidden_cond)
                visible_input_hidden_expect = self._backward(hid_suf_stat_mean)

                # sparsity
                if self.hidden_layer.hidden_sparsity == True and self.sparse_weight > 0:
                    hid_act = self.hidden_layer.get_hidden_activation(params_hidden_cond)
                    hg, wg, prev_hid_act = self._hidden_sparsity(sufficient_stat_batch,
                                                                 prev_hid_act, hid_act)
                    pos_hgrad += hg
                    pos_wgrad += wg

                hg, vg, wg = self._get_positive_grad(sufficient_stat_batch,
                                                     visible_input_hidden_expect,
                                                     hidden_input,
                                                     params_hidden_cond)

                pos_hgrad += hg
                pos_vgrad += vg
                pos_wgrad += wg

                # ======== sampling =========
                if self.learning_method == LEARNING_METHOD['cd']:
                    for icd in range(self.num_cd - 1):
                        suf_stat_hsample, visible_input, params_visible_cond, \
                        suf_stat_vsample, hidden_input, params_hidden_cond = \
                            self._gibbs_sampling(params_hidden_cond,
                                                 sampling=CD_SAMPLING['hidden_visible'])

                    suf_stat_hsample, visible_input, params_visible_cond, \
                    suf_stat_vsample, hidden_input, params_hidden_cond = \
                        self._gibbs_sampling(params_hidden_cond, sampling=self.sampling_in_last_cd)

                hid_suf_stat_mean = self.hidden_layer.hid_suf_stat_posterior_expectation(
                    params_hidden_cond)
                visible_input_hidden_expect = self._backward(hid_suf_stat_mean)

                # ======== negative phase =========
                neg_hgrad, neg_vgrad, neg_wgrad = self._get_negative_grad(
                    suf_stat_vsample, visible_input_hidden_expect, hidden_input, params_hidden_cond)

                # update params
                self.hgrad_inc = self.momentum * self.hgrad_inc \
                                 + self.learning_rate * (pos_hgrad - neg_hgrad)
                self.vgrad_inc = self.momentum * self.vgrad_inc \
                                 + self.learning_rate * (pos_vgrad - neg_vgrad)
                self.wgrad_inc = self.momentum * self.wgrad_inc \
                                 + self.learning_rate * (pos_wgrad - neg_wgrad
                                                         - self.weight_cost * self.w)

                self.h += self.hgrad_inc
                self.v += self.vgrad_inc
                self.w += self.wgrad_inc

                batch_logs.update(self._on_batch_end(x_batch, rdata=None))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid)
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _initialize_grad(self):
        pos_hgrad = np.zeros(self.h.shape)
        pos_vgrad = np.zeros(self.v.shape)
        pos_wgrad = np.zeros(self.w.shape)
        return pos_hgrad, pos_vgrad, pos_wgrad

    def _forward(self, sufficient_stat_visible, **kwargs):
        # Compute \sum_j feat_visible * w
        # Input:
        #   vsample :           #samples x # visible units x # visible sufficient statistics dim
        # Output:               # samples x # hidden units x # hidden sufficient statistics dim

        num_data = sufficient_stat_visible.shape[0]
        layer_input = np.zeros(
            [num_data, self.num_hidden,
             self.suf_stat_dim_hid])  # #samples x #dim x # hidden sufficient statistics dim
        coff = self.visible_layer.get_weight_coff(self.v)
        for i in range(self.suf_stat_dim_hid):
            S = np.zeros([num_data, self.num_hidden])
            for j in range(self.suf_stat_dim_vis):
                w_coff = np.multiply(self.w[:, :, j, i], coff[:, :, j].T)
                S += np.matmul(sufficient_stat_visible[:, :, j],
                               w_coff)  # #samples x # hidden units

            layer_input[:, :, i] = S
        return layer_input  # #samples x # hidden units x # hidden sufficient statistics dim

    def _backward(self, sufficient_stat_hidden, **kwargs):
        # compute \sum_j w * sufficient_stat_hidden
        # Input:
        #   sufficient_stat_hidden :    #samples x # hidden units x #hidden sufficient statistics dim
        # Output                        # samples x # visible units x #visible sufficient statistics dim

        num_data = sufficient_stat_hidden.shape[0]
        # #samples x #visible units x #visible sufficient statistics dim
        layer_input = np.zeros([num_data, self.num_visible, self.suf_stat_dim_vis])
        coff = self.hidden_layer.get_weight_coff(self.h)
        for i in range(self.suf_stat_dim_vis):
            S = np.zeros([num_data, self.num_visible])
            for j in range(self.suf_stat_dim_hid):
                w_coff = np.multiply(self.w[:, :, i, j].T, coff[:, :, j].T)
                S += np.matmul(sufficient_stat_hidden[:, :, j],
                               w_coff)  # #samples x # visible units
            layer_input[:, :, i] = S
        return layer_input  # #samples x # visible units x #visible sufficient statistics dim

    def _get_positive_grad(self, sufficient_stat_visible, visible_input,
                           hidden_input, params_hidden_cond):

        # #sample x #visible units x #visible suf
        num_data = sufficient_stat_visible.shape[0]
        vis_energy_expect = self.visible_layer.get_visible_energy_deri_expectation(
            sufficient_stat_visible,
            bias=self.v,
            other_layer_input=visible_input)

        # 1 x #visible units x #visible sufficient statistics dim
        vg = np.mean(vis_energy_expect, axis=0,
                     keepdims=True)

        # #sample x # hidden units x # hidden sufficient statistics dim
        hid_energy_expect = self.hidden_layer.get_hidden_energy_deri_expectation(
            sufficient_stat_visible=None,
            params_cond=params_hidden_cond,
            bias=self.h,
            other_layer_input=hidden_input)

        # 1 x #hidden units x # hidden sufficient statistics dim
        hg = np.mean(hid_energy_expect, axis=0,
                     keepdims=True)

        # # visible units x # hidden units x # visible suf x # hidden sufficient statistics dim
        wg = np.zeros(self.w.shape)

        # 1 x # visible units x  # visible sufficient stat dim
        visible_weight_coff = self.visible_layer.get_weight_coff(bias=self.v)
        # 1 x # hidden units x  # hidden sufficient stat dim
        hidden_weight_coff = self.hidden_layer.get_weight_coff(bias=self.h)

        expectation_hidden = self.hidden_layer.hid_suf_stat_posterior_expectation(
            params_cond=params_hidden_cond)
        for i in range(self.suf_stat_dim_vis):
            for j in range(self.suf_stat_dim_hid):
                # # visible units x # hidden units
                c = visible_weight_coff[:, :, i].T.dot(hidden_weight_coff[:, :, j])
                t = sufficient_stat_visible[:, :, i].T.dot(
                    expectation_hidden[:, :, j]) / num_data
                wg[:, :, i, j] = np.multiply(c, t)
        return hg, vg, wg

    def _get_negative_grad(self, sufficient_stat_vsample, visible_input,
                           hidden_input, params_hidden_cond):
        return self._get_positive_grad(sufficient_stat_vsample, visible_input,
                                       hidden_input, params_hidden_cond)

    def _gibbs_sampling(self, params_hidden_cond, sampling=CD_SAMPLING['hidden_visible']):
        if sampling == CD_SAMPLING['hidden']:
            sufficient_stat_hsample = self.hidden_layer.sample_sufficient_stat(params_hidden_cond)
            visible_input = self._backward(sufficient_stat_hsample)
            params_visible_cond = self.visible_layer.get_cond_dist_params(visible_input, self.v)

            sufficient_stat_vsample = self.visible_layer.hid_suf_stat_posterior_expectation(
                params_cond=params_visible_cond)

            hidden_input = self._forward(sufficient_stat_vsample)
            params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input, self.h)
        elif sampling == CD_SAMPLING['none']:
            sufficient_stat_hsample = self.hidden_layer.hid_suf_stat_posterior_expectation(
                params_cond=params_hidden_cond)

            visible_input = self._backward(sufficient_stat_hsample)
            params_visible_cond = self.visible_layer.get_cond_dist_params(visible_input, self.v)
            sufficient_stat_vsample = self.visible_layer.hid_suf_stat_posterior_expectation(
                params_cond=params_visible_cond)

            hidden_input = self._forward(sufficient_stat_vsample)
            params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input, self.h)
        elif sampling == CD_SAMPLING['hidden_visible']:
            sufficient_stat_hsample = self.hidden_layer.sample_sufficient_stat(
                params_cond=params_hidden_cond)
            visible_input = self._backward(sufficient_stat_hsample)
            params_visible_cond = self.visible_layer.get_cond_dist_params(visible_input,
                                                                          self.v)

            sufficient_stat_vsample = self.visible_layer.sample_sufficient_stat(
                params_cond=params_visible_cond)

            hidden_input = self._forward(sufficient_stat_vsample)
            params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input,
                                                                        self.h)
        else:  # visible sampling
            sufficient_stat_hsample = self.hidden_layer.hid_suf_stat_posterior_expectation(
                params_cond=params_hidden_cond)
            visible_input = self._backward(sufficient_stat_hsample)
            params_visible_cond = self.visible_layer.get_cond_dist_params(visible_input,
                                                                          self.v)

            sufficient_stat_vsample = self.visible_layer.sample_sufficient_stat(
                params_cond=params_visible_cond)

            hidden_input = self._forward(sufficient_stat_vsample)
            params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input,
                                                                        self.h)
        return sufficient_stat_hsample, visible_input, params_visible_cond, \
               sufficient_stat_vsample, hidden_input, params_hidden_cond

    def get_reconstruction(self, x):
        if len(x.shape) == 1:
            x = np.reshape(x, [1, x.shape[0]])
        sufficient_stat_x = self.visible_layer.get_sufficient_statistics_func(x)
        hidden_input = self._forward(sufficient_stat_x)
        params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input, self.h)

        hid_suf_stat_mean = self.hidden_layer.hid_suf_stat_posterior_expectation(params_hidden_cond)
        visible_input = self._backward(hid_suf_stat_mean)
        params_visible_cond = self.visible_layer.get_cond_dist_params(visible_input, self.v)
        vis_suf_stat_mean = self.visible_layer.hid_suf_stat_posterior_expectation(
            params_visible_cond)

        samples = self.visible_layer.sufficient_stat_to_value(vis_suf_stat_mean)
        return samples

    def transform(self, x):
        """Compute the mean of conditional probability,

        Parameters
        ----------
        x : {array-like, sparse matrix} shape (num_samples, num_visible)
            The data to be transformed.

        Returns
        -------~
        h : array, shape (num_samples, num_hidden)
            Latent representations of the data.
        """
        if len(x.shape) == 1:
            x = np.reshape(x, [1, x.shape[0]])

        sufficient_stat_x = self.visible_layer.get_sufficient_statistics_func(x)
        hidden_input = self._forward(sufficient_stat_x)
        params_hidden_cond = self.hidden_layer.get_cond_dist_params(hidden_input, self.h)

        hid_suf_stat_mean = self.hidden_layer.hid_suf_stat_posterior_expectation(
            params_hidden_cond)
        samples = self.hidden_layer.sufficient_stat_to_value(hid_suf_stat_mean)
        # suf_stats_samples = self.hidden_layer.sample_sufficient_stat(params_hidden_cond)
        # samples = self.hidden_layer.sufficient_stat_to_value(suf_stats_samples)
        return samples

    def get_free_energy(self, x):
        if len(x.shape) == 1:
            x = np.reshape(x, [1, x.shape[0]])

        num_data = x.shape[0]
        indi_params_visible = self.visible_layer.get_individual_params(self.v)
        indi_params_hidden = self.hidden_layer.get_individual_params(self.h)

        # #samples x #visible units x #sufficient stat dim
        sufficient_stat_x = self.visible_layer.get_sufficient_statistics_func(x)
        a = (np.log(self.visible_layer.get_additional_func(x))
             + np.multiply(np.tile(indi_params_visible, [num_data, 1, 1]), sufficient_stat_x))
        hidden_input = self._forward(sufficient_stat_x)

        delta_cond = indi_params_hidden + hidden_input
        b = self.hidden_layer.get_partition_func(delta_cond)  # #samples x #hidden units

        # vector of samples
        free_energy = -np.sum(a, axis=(1, 2)) - np.sum(b, axis=1)
        return np.reshape(free_energy, [num_data, 1])

    def _hidden_sparsity(self, sufficient_stat_x, prev_hid_act, hid_act):
        if prev_hid_act.shape == hid_act.shape:
            q = self.sparse_decay * prev_hid_act + (1 - self.sparse_decay) * hid_act
        else:
            q = (1 - self.sparse_decay) * hid_act
        prev_hid_act = np.copy(hid_act)
        sparse_grad = self.sparse_level - q
        hg = self.sparse_weight * np.mean(sparse_grad, axis=0, keepdims=True)
        wg = np.zeros(self.w.shape)
        for i in range(self.suf_stat_dim_vis):
            for j in range(self.suf_stat_dim_hid):
                wg[:, :, i, j] = self.sparse_weight * sufficient_stat_x[:, :, i].T.dot(
                    sparse_grad[:, :, j]) / sufficient_stat_x.shape[0]
        return hg, wg, prev_hid_act

    def display(self, param, **kwargs):
        if param == 'reconstruction':
            if 'ax' in kwargs:
                ax = kwargs['ax']
                data = kwargs['data']

                x_recon = self.get_reconstruction(data[:100, :])
                x_disp = np.zeros([100, data.shape[1]])

                for i in range(50):
                    x_disp[2 * i, :] = data[i, :]
                    x_disp[2 * i + 1, :] = x_recon[i, :]

                fig_recon = self._disp_images(x_disp, fig=ax.figure,
                                              title='Reconstruction')

    def _disp_images(self, x, fig=None, title="", img_shape=(28, 28),
                     tile_shape=(10, 10), block=False):

        img = tile_raster_images(x, img_shape=img_shape, tile_shape=tile_shape,
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=False)

        if fig is None:
            fig = plt.figure(title)
            plt.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none')
            plt.colorbar()
            plt.axis('off')
            plt.grid(False)
            plt.tight_layout()
        else:
            fig = plt.figure(fig.number)
            plt.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none')
            plt.axis('off')
            plt.grid(False)
            plt.title(title)

        # plt.show(block=block)
        return fig

    def get_params(self, deep=True):
        out = super(EFRBM, self).get_params(deep=deep)
        param_names = EFRBM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
