from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np
import numpy.random as np_rand
import scipy.stats as stats

from ... import Model
from ..distribution.gaussian_invwishart import GaussianInvWishart as GIW
from ...optimizers.newton import Newton


class BaNK(Model):
    """Bayesian Nonparametric Kernel
    """

    def __init__(self,
                 model_name="BaNK",
                 gamma=10,
                 rf_dim=400,
                 inner_regularization=1.0,
                 outer_regularization=1.0,
                 alpha=1.0,
                 kappa=0.1,
                 inner_max_loop=100,
                 max_outer_loop=50,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(BaNK, self).__init__(**kwargs)
        self.gamma = gamma  # kernel width
        self.rf_dim = rf_dim  # dim of random feature
        self.inner_regularization = inner_regularization  # regularization parameter
        self.outer_regularization = outer_regularization
        self.alpha = alpha  # prior for DPM
        self.kappa = kappa  # prior for GIW
        self.inner_max_loop = inner_max_loop
        self.outer_max_loop = max_outer_loop

    def _init(self):
        super(BaNK, self)._init()
        self.omega_ = None
        self.w_ = None
        self.newton_opt = Newton(learning_rate=0.8, tolerance=1e-7, max_loop=self.inner_max_loop)
        self.newton_opt.init_params(
            obj_func=BaNK.get_log_f,
            grad_func=BaNK.get_grad_log_f,
            hess_func=BaNK.get_hessian_log_f
        )

    @staticmethod
    def get_log_f(w, (phi, y, regular_param)):
        try:
            phi_beta = np.dot(phi, w)
            return \
                -np.dot(y, phi_beta) \
                + np.sum(np.log(1+np.exp(phi_beta))) \
                + 0.5 * regular_param * np.sum(w**2)
        except Exception:
            return 0

    @staticmethod
    def get_grad_log_f(w, (phi, y, regular_param)):
        try:
            sig_predict = (1.0 / (1 + np.exp(-(np.dot(phi, w)))))
            grad_pen = regular_param * w
        except FloatingPointError:
            print("Error")
        return np.dot(phi.T, (sig_predict - y)) + grad_pen

    @staticmethod
    def get_hessian_log_f(w, (phi, y, regular_param)):
        sig_predict = (1.0 / (1 + np.exp(-np.dot(phi, w))))
        return np.dot(phi.T * sig_predict * (1 - sig_predict), phi) + regular_param * np.eye(len(w))

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        num_samples = x.shape[0]
        input_dim = x.shape[1]
        rf_dim = self.rf_dim
        rf_2dim = rf_dim * 2
        rf_2dim_pad = rf_2dim + 1
        scale_rf = 1.0  # / np.sqrt(dim_rf)

        outer_max_loop = self.outer_max_loop

        self.x_ = x
        self.y_ = y

        kappa0 = self.kappa
        loc0 = np.zeros(input_dim)
        degree0 = input_dim + 3
        scale_mat0 = self.gamma * np.eye(input_dim, input_dim)

        num_kernels = int(rf_dim / 10)  # as in original code
        z = np.zeros(rf_dim).astype(int)
        omega = np.zeros((rf_dim, input_dim))

        for di in range(rf_dim):
            omega[di, :] = np_rand.multivariate_normal(
                np.zeros(input_dim), scale_mat0 / (degree0 - input_dim - 1))

        mean_lst = []
        cov_lst = []
        for k in range(num_kernels):
            mean_lst.append(np.zeros(input_dim))
            cov_lst.append(np.zeros((input_dim, input_dim)))

        nk = np.zeros(num_kernels)
        for di in range(rf_dim):
            zdi = z[di]
            nk[zdi] += 1

        w = np.zeros(rf_2dim_pad)
        omega_x = np.dot(omega, self.x_.T).T

        phi = np.ones((num_samples, rf_2dim_pad))
        phi[:, 0:rf_dim] = np.cos(omega_x)
        phi[:, rf_dim:rf_2dim] = np.sin(omega_x)
        phi *= scale_rf

        log_lap_beta_old = np.empty(rf_dim)
        log_pbeta_old = np.zeros(rf_dim)
        log_lap_beta_old[:] = np.nan
        log_mode_old = np.zeros(rf_dim)
        log_mode_old[:] = np.nan

        pp_mulgauss = np.zeros((num_kernels, rf_dim))
        log_alpha = np.nan

        w, _ = self.newton_opt.solve(w, (phi, self.y_, self.inner_regularization))
        # lap_matrix = BaNK.get_hessian_log_f(w, phi, self.y_, self.inner_regularization)

        for l in range(outer_max_loop):
            # sample mu, cov
            for k in range(num_kernels):
                mean_lst[k], cov_lst[k] = GIW.sample_posterior(
                    input_dim, omega[z == k, :], kappa0, loc0, degree0, scale_mat0)
                if nk[k] > 0:
                    print("mu = max:{}, min:{}".format(np.max(mean_lst[k]), np.min(mean_lst[k])))
                    print("cov = max:{}, min:{}".format(np.max(cov_lst[k]), np.min(cov_lst[k])))

            for di in range(rf_dim):
                for k in range(num_kernels):
                    pp_mulgauss[k, di] = stats.multivariate_normal. \
                        logpdf(omega[di, :], mean_lst[k], cov_lst[k])
            pp_mulgauss -= np.max(pp_mulgauss, axis=0)

            if np.isnan(log_alpha):
                log_alpha = np.percentile(np.abs(pp_mulgauss[pp_mulgauss < 0]), 1)
                print(log_alpha)

            # sample z
            for di in range(rf_dim):
                zdi = z[di]
                # remove
                nk[zdi] -= 1

                pp = pp_mulgauss[:, di]

                is_first = True
                for k in range(num_kernels):
                    if nk[k] > 0:
                        pp[k] += np.log(nk[k])
                    else:
                        if is_first:
                            pp[k] += log_alpha
                        else:
                            pp[k] = -np.inf
                        is_first = False
                try:
                    pp = np.exp(pp - np.max(pp))
                    pp /= np.sum(pp)
                    zdi_new = np.argmax(np.random.multinomial(1, pp))
                except FloatingPointError:
                    print("Error pp={}".format(pp))
                    zdi_new = np.argmax(pp)

                z[di] = zdi_new
                nk[zdi_new] += 1

            print("nk={}".format(nk[nk > 0]))

            if np.sum(np.abs(np.cos(np.dot(omega, self.x_.T).T) * scale_rf - phi[:, 0: rf_dim])) > 1e-7:
                print("Error Phi")

            # sample W & beta
            for di in range(rf_dim):
                di_idx = [di, di + rf_dim]
                n_di_idx = 2
                if di == rf_dim - 1:
                    di_idx = [di, di + rf_dim, rf_2dim]
                    n_di_idx = 3
                log_pbeta_old[di] = stats.multivariate_normal.\
                    logpdf(w[di_idx], np.zeros(n_di_idx), 1.0 / self.inner_regularization)

            n_accept = 0
            for di in range(rf_dim):
                di_idx = [di, di + rf_dim]
                n_di_idx = 2
                if di == rf_dim - 1:
                    di_idx = [di, di + rf_dim, rf_2dim]
                    n_di_idx = 3

                zdi = z[di]
                omega_di_old = omega[di, :].copy()
                phidi_old = phi[:, di_idx].copy()

                omega[di, :] = np.random.multivariate_normal(mean_lst[zdi], cov_lst[zdi])
                omega_x_di_tmp = np.dot(self.x_, omega[di, :].T)  # (N, 1)
                phi[:, di] = np.cos(omega_x_di_tmp) * scale_rf
                phi[:, di + rf_dim] = np.sin(omega_x_di_tmp) * scale_rf

                beta_idx_old = w[di_idx].copy()

                phi_beta_old = np.sum(phidi_old * beta_idx_old, axis=1)
                exp_phi_beta_old = np.exp(phi_beta_old)
                log_py_old = -np.sum(np.log(1.0 + exp_phi_beta_old)) + np.dot(phi_beta_old, self.y_)

                w_mode, _ = self.newton_opt.solve(
                    w[di_idx], (phi[:, di_idx], self.y_, self.inner_regularization))
                lap_matrix = BaNK.get_hessian_log_f(
                    w[di_idx], (phi[:, di_idx], self.y_, self.inner_regularization))

                lap_inv_matrix_a = np.linalg.inv(lap_matrix)
                w[di_idx] = np.multiply(lap_inv_matrix_a, np.random.normal(0, 1, (n_di_idx, 1))) + w_mode

                phi_w = np.sum(phi[:, di_idx] * w[di_idx], axis=1)
                exp_phi = np.exp(phi_w)
                log_py = -np.sum(np.log(1.0 + exp_phi)) + np.dot(phi_w, self.y_)
                try:
                    log_pbeta = stats.multivariate_normal.logpdf(w[di_idx], np.zeros(n_di_idx),
                                                                 1.0 / self.inner_regularization)
                except FloatingPointError:
                    print("Error log_pbeta = {}".format(w[di_idx]))

                log_mode = stats.multivariate_normal.logpdf(w[di_idx], w_mode, lap_inv_matrix_a)

                log_lap_beta = \
                    - (- 0.5 * np.log(np.abs(np.linalg.det(lap_matrix))))
                if np.isnan(log_lap_beta_old[di]):
                    log_lap_beta_old[di] = log_lap_beta
                if np.isnan(log_mode_old[di]):
                    log_mode_old[di] = log_mode

                accept_rate = min(0,
                                  ((log_py + log_pbeta + log_lap_beta_old[di] + log_mode_old[di]) -
                                   (log_py_old + log_pbeta_old[di] + log_lap_beta + log_mode)))

                if np.log(np.random.uniform()) > accept_rate:
                    w[di_idx] = beta_idx_old
                    omega[di, :] = omega_di_old
                    phi[:, di_idx] = phidi_old
                else:
                    log_lap_beta_old[di] = log_lap_beta
                    log_pbeta_old[di] = log_pbeta
                    log_mode_old[di] = log_mode
                    n_accept += 1

            phi_w = np.sum(phi * w, axis=1)
            print("err={}, accept={}".format(np.mean(self.y_ != (phi_w >= 0)), n_accept))

        w = np.zeros(rf_2dim_pad)
        w, _ = self.newton_opt.solve(w, (phi, self.y_, self.inner_regularization))

        self.omega_ = omega
        self.w_ = w

    def predict(self, x):
        rf_dim = self.rf_dim
        rf_2dim = rf_dim * 2
        rf_2dim_pad = rf_2dim + 1
        scale_rf = 1.0  # / np.sqrt(rf_dim)

        num_test = x.shape[0]
        y = np.ones(num_test, dtype=int)

        omega_x = np.dot(self.omega_, x.T).T
        phi = np.ones((num_test, rf_2dim_pad))
        phi[:, 0:rf_dim] = np.cos(omega_x)

        phi[:, rf_dim:rf_2dim] = np.sin(omega_x)
        phi *= scale_rf

        y[(np.sum(phi * self.w_, axis=1)) <= 0] = 0

        return self._decode_labels(y)

    def get_params(self, deep=True):
        out = super(BaNK, self).get_params(deep=deep)
        out.update({
            'gamma': self.gamma,
            'kappa': self.kappa,
            'inner_regularization': self.inner_regularization,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(BaNK, self).get_all_params(deep=deep)
        out.update(self.get_params(deep=deep))
        return out
