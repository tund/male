from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from collections import deque

from ...models.kernel.svrg import SVRG


class SVRG_S(SVRG):
    """ Stochastic Variance-reduced Gradient Descent for Kernel Online Learning
        Speedup Version
    """

    def __init__(self,
                 model_name="SVRG_S",
                 **kwargs):
        super(SVRG_S, self).__init__(model_name=model_name, **kwargs)

    def _get_grad_full_pre_calc_rf(self, xn, yn, n, kn):
        xn_rf = self.x_rf[n, :]
        yn_pred, wxn, wxn_rf, wxn_core, kn = self._predict_one_given_w_pre_calc_rf_kn(
            xn, self.w_full_rf, self.w_full_core, self.num_core, xn_rf, kn
        )
        return self._get_grad(wxn, yn), kn

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        np.random.seed()  # BUG should check the whole framework

        self.x_ = x
        self.y_ = y

        num_samples = x.shape[0]
        input_dim = x.shape[1]
        rf_dim = self.rf_dim
        rf_2dim_pad = self.rf_2dim_pad

        if self.verbose > 0:
            print('Speedup Version')

        # construct random features
        self.omega = np.random.normal(0, self.gamma / 2.0, (input_dim, rf_dim))
        omega_x = np.matmul(x, self.omega)  # (num_samples, rf_dim)
        self.x_rf = self.rf_scale * np.hstack((np.cos(omega_x), np.sin(omega_x), np.ones((num_samples, 1))))

        self.w_cur_core = np.zeros((self.num_classes, num_samples))
        self.idx_core = np.zeros(num_samples, dtype=int)
        self.chk_core = -np.ones(num_samples, dtype=int)
        self.num_core = 0

        self.w_cur_rf = np.zeros((self.num_classes, rf_2dim_pad))

        self.w_full_core = np.zeros((self.num_classes, num_samples))
        self.w_full_rf = np.zeros((self.num_classes, rf_2dim_pad))

        self.w_cur_core[0] = 0
        self.w_full_core[0] = 0
        self.idx_core[0] = 0
        self.chk_core[0] = 0
        self.num_core += 1

        grad_lst = deque()
        xnt_rf_lst = deque()
        grad_idx_lst = deque()
        k_lst = deque()

        sum_grad_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
        sum_grad_core = np.zeros((self.num_classes, num_samples))
        w_sum_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
        w_sum_core = np.zeros((self.num_classes, num_samples))

        max_loop = self.num_epochs * num_samples
        move_decision = np.zeros(max_loop)
        idx_samples = np.zeros(max_loop, dtype=int)
        for n in range(max_loop):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)
            if (n % self.freq_calc_metrics) == 0:
                if 'train_loss' in callback_metrics:
                    mean_loss = self._get_mean_loss(x, y)
                    epoch_logs['train_loss'] = mean_loss
                if 'valid_loss' in callback_metrics:
                    y_valid_trans = self._transform_labels(y_valid)
                    mean_loss = self._get_mean_loss(x_valid, y_valid_trans)
                    epoch_logs['valid_loss'] = mean_loss
                if 'obj_func' in callback_metrics:
                    mean_loss = self._get_mean_loss(x, y)
                    dual_wnorm2 = self._get_dual_wnorm2()
                    obj_func = self.regular_param * dual_wnorm2 + mean_loss
                    epoch_logs['obj_func'] = obj_func

            nt = np.random.randint(num_samples)
            xnt = x[nt, :]
            ynt = y[nt]
            idx_samples[n] = nt

            # predict
            ynt_pred, wxnt, xnt_rf, wxnt_rf, dist2_xnt, knt, wxnt_core = self._predict_one_pre_calc_rf(xnt, nt)
            grad_cur, idx_cur_runner, loss_cur = self._get_grad(wxnt, ynt)
            (grad_full, idx_full_runner, loss_full), knt = self._get_grad_full_pre_calc_rf(xnt, ynt, nt, knt)

            move_decision[n] = {
                SVRG_S.BUDGET: self._oracle_budget,
                SVRG_S.COVERAGE: self._oracle_coverage,
                SVRG_S.ALWAYS_MOVE: self._oracle_always,
            }[self.oracle](dist2_xnt, n)

            if len(grad_lst) > self.cache_size - 1:
                xnt_rf_lst.popleft()
                idx_pop = grad_idx_lst.popleft()
                k_lst.popleft()
                grad_tmp = grad_lst.popleft()

                if len(grad_tmp.shape) == 2:
                    sum_grad_rf -= grad_tmp
                else:
                    sum_grad_core[:, self.chk_core[idx_pop]] -= grad_tmp

            xnt_rf_lst.append(xnt_rf)
            grad_idx_lst.append(nt)
            k_lst.append(knt)

            if move_decision[n]:
                # approximate
                vnt_rf_cur = np.zeros((self.num_classes, self.rf_2dim_pad))
                vnt_rf_cur[ynt, :] = grad_cur * xnt_rf
                vnt_rf_cur[idx_cur_runner, :] = -grad_cur * xnt_rf

                vnt_rf_full = np.zeros((self.num_classes, self.rf_2dim_pad))
                vnt_rf_full[ynt, :] = grad_full * xnt_rf
                vnt_rf_full[idx_full_runner, :] = -grad_full * xnt_rf

                sum_grad_rf += vnt_rf_full
                grad_lst.append(vnt_rf_full)

                vnt = vnt_rf_cur - vnt_rf_full + sum_grad_rf / len(grad_lst)
                self.w_cur_rf = \
                    (self.w_cur_rf - self.learning_rate * vnt) / \
                    (self.learning_rate * self.regular_param + 1)

                self.w_cur_core[:, :self.num_core] += \
                    - sum_grad_core[:, :self.num_core] * self.learning_rate \
                    / (len(grad_lst) * (self.learning_rate * self.regular_param + 1))
            else:
                # add to core
                vnt_core_cur = np.zeros(self.num_classes)
                vnt_core_cur[ynt] = grad_cur
                vnt_core_cur[idx_cur_runner] = -grad_cur

                vnt_core_full = np.zeros(self.num_classes)
                vnt_core_full[ynt] = grad_full
                vnt_core_full[idx_full_runner] = -grad_full

                if self.chk_core[nt] < 0:
                    self.num_core += 1
                    self.idx_core[self.num_core - 1] = nt
                    self.chk_core[nt] = self.num_core - 1
                    self.w_cur_core[:, self.num_core - 1] = np.zeros(self.num_classes)

                sum_grad_core[:, self.chk_core[nt]] += vnt_core_full
                grad_lst.append(vnt_core_full)

                self.w_cur_core[:, self.chk_core[nt]] += \
                    - (vnt_core_cur - vnt_core_full) * self.learning_rate \
                    / (self.learning_rate * self.regular_param + 1)
                # CARE when upgrade to BATCH SETTING += NOT =

                self.w_cur_core[:, :self.num_core] += \
                    - sum_grad_core[:, :self.num_core] * self.learning_rate \
                    / (len(grad_lst) * (self.learning_rate * self.regular_param + 1))

                self.w_cur_rf += -sum_grad_rf * self.learning_rate \
                    / (len(grad_lst) * (self.learning_rate * self.regular_param + 1))

            w_sum_rf += self.w_cur_rf
            w_sum_core[:, :self.num_core] += self.w_cur_core[:, :self.num_core]

            if (n+1) % self.freq_update_full_model == 0:
                self.w_full_rf = w_sum_rf / self.freq_update_full_model
                # self.w_cur_rf = self.w_full_rf.copy()
                w_sum_rf = np.zeros((self.num_classes, self.rf_2dim_pad))

                self.w_full_core = w_sum_core / self.freq_update_full_model
                # self.w_cur_core = self.w_full_core.copy()
                w_sum_core = np.zeros((self.num_classes, num_samples))

                sum_grad_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
                sum_grad_core = np.zeros((self.num_classes, num_samples))
                grad_lst.clear()

                for i in range(n-self.cache_size+1, n+1):
                    if i < 0:
                        continue
                    it = idx_samples[i]

                    xit_tmp = x[it, :]
                    yit_tmp = y[it]
                    kit_tmp = k_lst.popleft()
                    xit_rf_tmp = xnt_rf_lst.popleft()
                    (grad_full, idx_full_runner, loss_full), kit_tmp = self._get_grad_full_pre_calc_rf(
                        xit_tmp, yit_tmp, it, kit_tmp)

                    k_lst.append(kit_tmp)
                    xnt_rf_lst.append(xit_rf_tmp)

                    if self.chk_core[it] < 0:
                        vit_rf_full = np.zeros((self.num_classes, self.rf_2dim_pad))
                        vit_rf_full[ynt, :] = grad_full * self.x_rf[it, :]
                        vit_rf_full[idx_full_runner, :] = -grad_full * self.x_rf[it, :]

                        sum_grad_rf += vit_rf_full
                        grad_lst.append(vit_rf_full)
                    else:
                        vit_core_full = np.zeros(self.num_classes)
                        vit_core_full[yit_tmp] = grad_full
                        vit_core_full[idx_full_runner] = -grad_full

                        sum_grad_core[:, self.chk_core[it]] += vit_core_full
                        grad_lst.append(vit_core_full)

            if (n % self.freq_calc_metrics) == 0:
                self.epoch += 1
                callbacks.on_epoch_end(self.epoch, epoch_logs)
        if self.verbose > 0:
            print('num_core=', self.num_core)
        self.w_cur_core = self.w_full_core.copy()
        self.w_cur_rf = self.w_cur_rf.copy()

    def _predict_one(self, xn):
        return self._predict_one_given_w(self.w_cur_rf, self.w_cur_core, self.num_core, xn)

    def _predict_one_pre_calc_rf(self, xn, n):
        xn_rf = self.x_rf[n, :]
        return self._predict_one_given_w_pre_calc_rf(self.w_cur_rf, self.w_cur_core, self.num_core, xn, xn_rf)

    def get_params(self, deep=True):
        out = super(SVRG_S, self).get_params(deep=deep)
        param_names = SVRG_S._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
