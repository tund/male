from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from ... import Model
from ...utils.disp_utils import visualize_classification_prediction
from ...utils.label import LabelEncoderDict


class GKM(Model):
    """ Graph-based Kernel Machine
    """
    HINGE = 0
    L1 = 1
    LOGISTIC = 2
    INSENSITIVE = 3
    S_HINGE = 4
    UNLABEL = -128

    def __init__(self,
                 model_name="GKM",
                 mode='batch',
                 unlabel=-128,
                 trade_off_1=0.1,
                 trade_off_2=0.1,
                 gamma=10,
                 loss_func=HINGE,
                 smooth_hinge_theta=0.5,
                 smooth_hinge_tau=0.5,
                 insensitive_epsilon=0.001,
                 unlabel_loss_func_degree=2,
                 sim_func=None,
                 sim_params=(1.0, 0),
                 freq_calc_metrics=1,
                 **kwargs):
        super(GKM, self).__init__(model_name=model_name, **kwargs)
        self.mode = mode
        self.unlabel = unlabel
        self.trade_off_1 = trade_off_1
        self.trade_off_2 = trade_off_2
        self.gamma = gamma
        self.loss_func = loss_func
        self.smooth_hinge_theta = smooth_hinge_theta
        self.smooth_hinge_tau = smooth_hinge_tau
        self.insensitive_epsilon = insensitive_epsilon
        self.unlabel_loss_func_degree = unlabel_loss_func_degree
        self.sim_func = sim_func
        self.sim_params = sim_params
        self.freq_calc_metrics = freq_calc_metrics

    def _init(self):
        super(GKM, self)._init()

        self.w = None
        self.wbar = None
        self.bias = None
        self.idx_cores = None
        self.num_cores = None
        self.idx_data_cores = None

        self.encoded_unlabel = None

        self.selected_indices = None

        if self.sim_func is None:
            self.sim_func = GKM.calc_similarity_kernel
        if self.sim_params is None:
            self.sim_params = self.gamma

    @staticmethod
    def calc_similarity_1d(x, y, params):
        scale, idx_feature = params
        return np.exp(-scale * np.abs(x[idx_feature]-y[idx_feature]))

    @staticmethod
    def calc_similarity_kernel(x, y, params):
        gamma = params
        return np.exp(-gamma * np.sum((x - y) ** 2))

    def _get_loss_label(self, wx, ywx, y):
        loss = 0
        if self.loss_func == GKM.HINGE:
            loss = -y if (ywx <= 1) else 0
        elif self.loss_func == GKM.L1:
            loss = +1 if (wx - y >= 0) else -1
        elif self.loss_func == GKM.LOGISTIC:
            if ywx < -500:
                loss = -y
            elif ywx > 500:
                loss = 0
            else:
                tmp = np.exp(-ywx)
                loss = -y * tmp / (tmp + 1)
        elif self.loss_func == GKM.INSENSITIVE:
            if np.abs(y - wx) > self.insensitive_epsilon:
                loss = +1 if (wx - y > 0) else -1
        elif self.loss_func == GKM.S_HINGE:
            if ywx < 1 - self.smooth_hinge_tau:
                loss = -y
            elif ywx <= 1:
                loss = (ywx - 1) * y / self.smooth_hinge_tau
        else:
            raise ValueError('Incorrect loss function')
        return loss

    def _get_dist2(self, xn):
        dist2 = np.sum(
            (self.x_[self.idx_cores[:self.num_cores], :] - xn) ** 2, axis=1)
        return dist2

    def _get_kernel(self, xn):
        dist2 = self._get_dist2(xn)
        kn = np.exp(-self.gamma * dist2)
        return kn

    def _get_wx(self, xn):
        kn = self._get_kernel(xn)
        wxn = np.sum(self.w[:self.num_cores] * kn)
        return wxn

    def _get_wbarx(self, xn):
        kn = self._get_kernel(xn)
        wxn = np.sum(self.wbar[:self.num_cores] * kn)
        return wxn

    def _get_mean_loss(self, x, y):
        num_tests = x.shape[0]
        mean_loss = 0.0
        for nn in range(num_tests):
            xn = x[nn, :]
            yn = y[nn]
            if yn == GKM.UNLABEL:
                continue
            yn_pred, wxn = self._predict_one(xn)
            loss_cur = self._get_loss_label(wxn, yn * wxn, yn)
            mean_loss += np.maximum(0, loss_cur)
        mean_loss = mean_loss / num_tests
        return mean_loss

    def _get_mean_loss_unlabel(self, x, y):
        num_tests = x.shape[0]
        mean_loss = 0.0
        for n1 in range(num_tests):
            xn1 = x[n1, :]
            wxn1 = self._get_wx(xn1)
            n2 = np.random.randint(0, num_tests)
            xn2 = x[n2, :]
            wxn2 = self._get_wx(xn2)

            wxn12 = wxn1 - wxn2
            mu = self.sim_func(xn1, xn2, self.sim_params)

            loss_cur = mu * (np.abs(wxn12) ** self.unlabel_loss_func_degree)
            mean_loss += np.maximum(0, loss_cur)
        mean_loss = mean_loss / num_tests
        return mean_loss

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        np.random.seed()

        self.x_ = x
        self.y_ = y

        np.seterr(under='ignore')

        num_samples = self.x_.shape[0]

        # process unlabel data
        self.encoded_unlabel = super(GKM, self)._transform_labels([self.unlabel])[0]
        self.num_classes -= 1
        if self.num_classes > 2:
            raise ValueError('Not support multi-class')
        if self.encoded_unlabel == 0:
            self.encoded_poslabel = 2
            self.encoded_neglabel = 1
        elif self.encoded_unlabel == 1:
            self.encoded_poslabel = 2
            self.encoded_neglabel = 0
        else:
            self.encoded_poslabel = 1
            self.encoded_neglabel = 0

        self.y_[self.y_ == self.encoded_unlabel] = GKM.UNLABEL
        self.y_[self.y_ == self.encoded_neglabel] = -1
        self.y_[self.y_ == self.encoded_poslabel] = 1

        self.label_encoder = LabelEncoderDict(self.label_encoder)
        self.label_encoder.encoded_label[self.encoded_poslabel] = 1
        self.label_encoder.encoded_label[self.encoded_neglabel] = -1
        self.label_encoder.encoded_label[self.encoded_unlabel] = GKM.UNLABEL
        self.label_encoder.refresh_dict()

        self.idx_unlabel = np.arange(num_samples)[self.y_ == GKM.UNLABEL]
        self.idx_label = np.arange(num_samples)[self.y_ != GKM.UNLABEL]
        self.num_unlabel = len(self.idx_unlabel)
        self.num_label = len(self.idx_label)

        if self.verbose > 0:
            print('num_labels:', self.num_label)
            print('num_unlabel', self.num_unlabel)

        trade_off_1 = 1.0 * self.trade_off_1 * self.num_label
        trade_off_2 = 1.0 * self.trade_off_2 * self.num_label

        max_loop = self.num_epochs * num_samples

        self.w = np.zeros(num_samples)
        self.wbar = np.zeros(num_samples)
        self.idx_cores = np.zeros(num_samples, dtype=int)
        self.idx_data_cores = -np.ones(num_samples, dtype=int)
        self.num_cores = 0

        updated_lst = self.idx_label.copy()

        self.w[0] = 0
        self.wbar[0] = 0
        self.idx_cores[0] = 0
        self.idx_data_cores[0] = 0
        self.num_cores += 1

        self.bias = 0

        self.selected_indices = np.zeros(3, dtype=int)

        for t in range(1, max_loop):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)
            if (t % self.freq_calc_metrics) == 0:
                if 'train_loss' in callback_metrics:
                    mean_loss = self._get_mean_loss_unlabel(self.x_, self.y_)
                    epoch_logs['train_loss'] = mean_loss

            eta = 2.0 / (t + 1.0)

            it = self.idx_label[np.random.randint(0, self.num_label)]
            ut = updated_lst[np.random.randint(0, len(updated_lst))]
            vt = self.idx_unlabel[np.random.randint(0, self.num_unlabel)]

            self.selected_indices[0] = it
            self.selected_indices[1] = ut
            self.selected_indices[2] = vt

            wxit = self._get_wx(self.x_[it, ])
            ywxit = wxit * self.y_[it]

            wxut = self._get_wx(self.x_[ut, :])
            wxvt = self._get_wx(self.x_[vt, :])
            wxuv = wxut - wxvt

            mu = self.sim_func(self.x_[ut, :], self.x_[vt, :], self.sim_params)

            self.w -= self.w * eta

            loss_label = self._get_loss_label(wxit, ywxit, self.y_[it])
            if loss_label != 0:
                if self.idx_data_cores[it] < 0:
                    self.idx_data_cores[it] = self.num_cores
                    self.idx_cores[self.num_cores] = it
                    self.w[self.num_cores] = 0
                    self.wbar[self.num_cores] = 0
                    self.num_cores += 1

                self.w[self.idx_data_cores[it]] -= trade_off_1 * eta * loss_label

            loss_unlabel = \
                self.unlabel_loss_func_degree * np.sign(wxuv) * \
                np.power(np.abs(wxuv), self.unlabel_loss_func_degree - 1)
            if mu * loss_unlabel != 0:
                if self.idx_data_cores[ut] < 0:
                    self.idx_data_cores[ut] = self.num_cores
                    self.idx_cores[self.num_cores] = ut
                    self.w[self.num_cores] = 0
                    self.wbar[self.num_cores] = 0
                    self.num_cores += 1
                if self.idx_data_cores[vt] < 0:
                    self.idx_data_cores[vt] = self.num_cores
                    self.idx_cores[self.num_cores] = vt
                    self.w[self.num_cores] = 0
                    self.wbar[self.num_cores] = 0
                    self.num_cores += 1
                    np.append(updated_lst, vt)

                if self.verbose > 0:
                    print('Update unlabel')
                # print(trade_off_2 * eta * mu * loss_unlabel)
                # print(self.w)
                self.w[self.idx_data_cores[ut]] -= trade_off_2 * eta * mu * loss_unlabel
                self.w[self.idx_data_cores[vt]] += trade_off_2 * eta * mu * loss_unlabel

            self.wbar = (t - 1.0) / (t + 1.0) * self.wbar + eta * self.w

            if (t % self.freq_calc_metrics) == 0:
                self.epoch += 1
                callbacks.on_epoch_end(self.epoch, epoch_logs)

        # self.y_[self.y_ == -1] = 0  # convert to zero index

    def _predict_one(self, xn):
        wx = self._get_wbarx(xn) - self.bias
        return +1 if wx > 0 else -1, wx

    def predict(self, x):
        y = np.zeros(x.shape[0], dtype=int)
        for n in range(x.shape[0]):
            y[n], _ = self._predict_one(x[n])

        # y[y == -1] = 0  # convert to zero index
        y = self.label_encoder.inverse_transform(y)
        return y

    def display_prediction(self, **kwargs):
        kwargs['selected_indices'] = self.selected_indices
        visualize_classification_prediction(
            self, self.x_, self.y_, **kwargs)

    def display(self, param, **kwargs):
        if param == 'predict':
            self.display_prediction(**kwargs)
        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        out = super(GKM, self).get_params(deep=deep)
        param_names = GKM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
