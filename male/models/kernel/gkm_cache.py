from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from ... import Model
from ...utils.disp_utils import visualize_classification_prediction
from ...utils.label import LabelEncoderDict
from ...models.kernel.gkm import GKM

from cachetools import LRUCache


class CacheGKM(GKM):
    """ Graph-based Kernel Machine
    """
    HINGE = 0
    L1 = 1
    LOGISTIC = 2
    INSENSITIVE = 3
    S_HINGE = 4
    UNLABEL = -128
    CACHE_SIZE = 10

    def __init__(self,
                 model_name="GKM_CACHE",
                 cache_size=100,
                 **kwargs):
        super(CacheGKM, self).__init__(model_name=model_name, **kwargs)
        self.cache_size = cache_size

    def _init(self):
        super(CacheGKM, self)._init()

    @staticmethod
    def calc_similarity_1d(x, y, params):
        scale, idx_feature = params
        return np.exp(-scale * np.abs(x[idx_feature]-y[idx_feature]))

    @staticmethod
    def calc_similarity_kernel(x, y, params):
        gamma = params
        return np.exp(-gamma * np.sum((x - y) ** 2))

    def _get_dist2(self, xn, start=0):
        dist2 = np.sum(
            (self.x_[self.idx_cores[start:self.num_cores], :] - xn) ** 2, axis=1)
        return dist2

    def _get_wx(self, xn, n=None):
        if n is not None:
            kn = self.cache.get(n, default=np.empty(0))
            len_kn = len(kn)
            if len_kn < self.num_cores:
                kn_tmp = self._get_kernel(xn, len_kn)
                kn = np.concatenate([kn, kn_tmp])
                self.cache.update[n] = kn
        else:
            kn = self._get_kernel(xn)

        wxn = np.sum(self.w[:self.num_cores] * kn)
        return wxn

    def _get_wbarx(self, xn):
        kn = self._get_kernel(xn)
        wxn = np.sum(self.wbar[:self.num_cores] * kn)
        return wxn

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
        self.encoded_unlabel = super(CacheGKM, self)._transform_labels([self.unlabel])[0]
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

        self.y_[self.y_ == self.encoded_unlabel] = CacheGKM.UNLABEL
        self.y_[self.y_ == self.encoded_neglabel] = -1
        self.y_[self.y_ == self.encoded_poslabel] = 1

        self.label_encoder = LabelEncoderDict(self.label_encoder)
        self.label_encoder.encoded_label[self.encoded_poslabel] = 1
        self.label_encoder.encoded_label[self.encoded_neglabel] = -1
        self.label_encoder.encoded_label[self.encoded_unlabel] = CacheGKM.UNLABEL
        self.label_encoder.refresh_dict()

        self.idx_unlabel = np.arange(num_samples)[self.y_ == CacheGKM.UNLABEL]
        self.idx_label = np.arange(num_samples)[self.y_ != CacheGKM.UNLABEL]
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

        self.selected_indices = np.zeros(3,dtype=int)

        self.cache = LRUCache(maxsize=self.cache_size)

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
                self.w[self.idx_data_cores[ut]] -= trade_off_2 * eta * mu * loss_unlabel
                self.w[self.idx_data_cores[vt]] += trade_off_2 * eta * mu * loss_unlabel

            self.wbar = (t - 1.0) / (t + 1.0) * self.wbar + eta * self.w

            if (t % self.freq_calc_metrics) == 0:
                self.epoch += 1
                callbacks.on_epoch_end(self.epoch, epoch_logs)
