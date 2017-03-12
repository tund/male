from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np
import scipy.stats as stats

from ... import Model


class BANK(Model):
    """BAyesian Nonparametric Kernel
    """

    def __init__(self,
                 model_name="BANK",
                 gamma=10,
                 dim_rf=400,
                 lbd=1.0,
                 alpha=1.0,
                 kappa=0.1,
                 inner_epoch=1,
                 max_loop=50,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(BANK, self).__init__(**kwargs)
        self.gamma = gamma  # kernel width
        self.dim_rf = dim_rf  # dim of random feature
        self.lbd = lbd  # regularization parameter
        self.alpha = alpha  # prior for DPM
        self.kappa = kappa  # prior for GIW
        self.inner_epoch = inner_epoch
        self.max_loop = max_loop

    def _init(self):
        super(BANK, self)._init()
        self.W_ = None
        self.beta_ = None

    @staticmethod
    def get_prior_giw(D, W, kappa, mu, nu, Psi):
        # W just contain W_di have Z_di = k
        if len(W.shape) < 2:
            raise Exception('wrong dimension')

        nk = W.shape[0]
        kappa_mu = kappa * mu
        if nk > 0:
            xbar = np.mean(W, axis=0)  # (D,)
            W_xbar = W - xbar  # (D,nk)
            C = np.dot(W_xbar.T, W_xbar)  # (D,D)
        else:
            xbar = np.zeros(D)
            C = np.zeros((D, D))
        mu_new = (kappa_mu + nk * xbar) / (kappa + nk)
        nu_new = nu + nk
        xbar_mu = (xbar - mu).reshape((D, 1))
        Psi_new = Psi + C + (kappa * nk / (kappa + nk)) * np.dot(xbar_mu, xbar_mu.T)
        cov_new = Psi_new / (nu_new - D - 1)
        return [mu_new, cov_new]

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        N = x.shape[0]
        D = x.shape[1]
        dim_rf = self.dim_rf
        dprime = dim_rf * 2
        dprime_ext = dprime + 1
        scale_rf = 1.0 / np.sqrt(dim_rf)

        L = self.max_loop

        self.x_ = x
        self.y_ = y
        # tempt fix bug
        self.y_[self.y_ == 0] = -1

        kappa0 = self.kappa
        mu0 = np.zeros(D)
        nu0 = D + 3
        Psi0 = self.gamma * np.eye(D, D)

        K = int(dim_rf / 10)  # as in original code
        Z = np.zeros(dim_rf).astype(int)
        W = np.zeros((dim_rf, D))

        mu_lst = []
        cov_lst = []
        for k in range(K):
            mu_lst.append(np.zeros(D))
            cov_lst.append(np.zeros((D, D)))
        nk = np.zeros(K)
        for di in range(dim_rf):
            nk[Z[di]] += 1
        for k in range(K):
            mu_lst[k], cov_lst[k] = BANK.get_prior_giw(D, W[Z == k, :], kappa0, mu0, nu0, Psi0)

        for di in range(dim_rf):
            W[di, :] = np.random.multivariate_normal(mu_lst[Z[di]], cov_lst[Z[di]])

        beta = np.zeros(dprime_ext)
        # for di in range(dim_rf):
        #     di_idx = [di, di + dim_rf]
        #     n_di_idx = 2
        #     if di == dim_rf - 1:
        #         di_idx = [di, di + dim_rf, dprime]
        #         n_di_idx = 2
        #     beta[di_idx] = stats.multivariate_normal.rvs(np.zeros(n_di_idx), 1.0 / self.lbd)
        Wx = np.dot(W, self.x_.T).T

        Phi = np.ones((N, dprime_ext))
        Phi[:, 0:dim_rf] = np.cos(Wx)
        Phi[:, dim_rf:dprime] = np.sin(Wx)
        Phi *= scale_rf

        log_lap_beta_old = np.zeros(dim_rf)
        log_pbeta_old = np.zeros(dim_rf)

        for l in range(L):
            # print("W={}".format(W))
            # sample mu, cov
            for k in range(K):
                mu_lst[k], cov_lst[k] = BANK.get_prior_giw(D, W[Z == k, :], kappa0, mu0, nu0, Psi0)
                if nk[k] > 0:
                    print("mu = {}".format(mu_lst[k]))
                    print("cov = {}".format(cov_lst[k]))

            # sample Z
            for di in range(dim_rf):
                zdi = Z[di]
                # remove
                nk[zdi] -= 1

                pp = np.zeros(K)
                for k in range(K):
                    pp[k] = stats.multivariate_normal. \
                        logpdf(W[di, :], mu_lst[k], cov_lst[k])
                    if pp[k] < -300:
                        print(pp[k])
                pp -= np.max(pp)

                for k in range(K):
                    if nk[k] > 0:
                        pp[k] += np.log(nk[k])
                    else:
                        pp[k] += np.log(self.alpha)
                try:
                    pp = np.exp(pp - np.max(pp))
                    pp /= np.sum(pp)
                    zdi_new = np.argmax(np.random.multinomial(5, pp))
                except FloatingPointError:
                    print("Error pp={}".format(pp))
                    zdi_new = np.argmax(pp)

                Z[di] = zdi_new
                nk[zdi_new] += 1

            # sample W & beta
            exp_PhiXy = np.exp(np.sum(-Phi * beta, axis=1) * self.y_)  # (N,)
            log_py_old = -np.sum(np.log(1.0 + exp_PhiXy))
            print("loss={}".format(-log_py_old))

            for di in range(dim_rf):
                di_idx = [di, di + dim_rf]
                n_di_idx = 2
                if di == dim_rf - 1:
                    di_idx = [di, di + dim_rf, dprime]
                    n_di_idx = 3
                log_pbeta_old[di] = stats.multivariate_normal.logpdf(beta[di_idx], np.zeros(n_di_idx), 1.0 / self.lbd)
                lap_matrix_a = self.lbd * np.eye(n_di_idx)  # np.zeros((2, 2))
                for n in range(N):
                    phi_pair = Phi[n:1, di_idx]
                    lap_matrix_a += np.dot(phi_pair.T, phi_pair) * (exp_PhiXy[n] / (exp_PhiXy[n]) ** 2)
                log_lap_beta_old[di] = (0.5 * N * np.log(2 * np.pi)) \
                                       + (- 0.5 * np.log(np.abs(np.linalg.det(lap_matrix_a))))

            for di in range(dim_rf):
                di_idx = [di, di + dim_rf]
                n_di_idx = 2
                if di == dim_rf - 1:
                    di_idx = [di, di + dim_rf, dprime]
                    n_di_idx = 3

                zdi = Z[di]
                Wdi_old = W[di, :]
                Phidi_old = Phi[:, di_idx]
                W[di, :] = np.random.multivariate_normal(mu_lst[zdi], cov_lst[zdi])
                Wx_di_tmp = np.dot(self.x_, W[di, :].T)  # (N, 1)
                Phi[:, di] = np.cos(Wx_di_tmp) * scale_rf
                Phi[:, di + dim_rf] = np.sin(Wx_di_tmp) * scale_rf

                T = self.inner_epoch * N
                beta_idx_old = beta[di_idx]
                for t in range(T):
                    idx = np.random.randint(0, N, self.batch_size)  # (t,)
                    PhiX_t = np.sum(Phi[idx, :] * beta, axis=1)  # (t,)
                    PhiXy_t = PhiX_t * self.y_[idx]  # (t,)
                    exp_minus_PhiXy_t = np.exp(-PhiXy_t)
                    grad = self.lbd - self.y_[idx] * exp_minus_PhiXy_t / (exp_minus_PhiXy_t + 1)
                    # beta[di_idx] *= (1.0 * t) / (t + 1)
                    beta[di_idx] -= \
                        (1.0 / (self.lbd * (t + 1))) * np.sum(Phi[idx, :][:, di_idx].T * grad, axis=1) / self.batch_size

                exp_PhiXy = np.exp(np.sum(-Phi * beta, axis=1) * self.y_)
                log_py = -np.sum(np.log(1.0 + exp_PhiXy))
                try:
                    log_pbeta = stats.multivariate_normal.logpdf(beta[di_idx], np.zeros(n_di_idx), 1.0 / self.lbd)
                except FloatingPointError:
                    print("Error log_pbeta = {}".format(beta[di_idx]))
                lap_matrix_a = self.lbd * np.eye(n_di_idx)  # np.zeros((2, 2))
                for n in range(N):
                    phi_pair = Phi[n:1, di_idx]
                    lap_matrix_a += np.dot(phi_pair.T, phi_pair) * (exp_PhiXy[n] / (exp_PhiXy[n]) ** 2)
                log_lap_beta = \
                    (0.5 * N * np.log(2 * np.pi)) \
                    + (- 0.5 * np.log(np.abs(np.linalg.det(lap_matrix_a))))
                try:
                    accept_rate = min(1, np.exp((log_py + log_pbeta + log_lap_beta_old[di]) -
                                                (log_py_old + log_pbeta_old[di] + log_lap_beta)))
                except FloatingPointError:
                    print("Error:{},{},{} {},{},{}".format(log_py, log_pbeta, log_lap_beta_old[di],
                                                           log_py_old, log_pbeta_old[di], log_lap_beta))
                print("Accept 1:{},{},{} {},{},{}".format(log_py, log_pbeta, log_lap_beta_old[di],
                                                          log_py_old, log_pbeta_old[di], log_lap_beta))
                print("Loss old={}, new={}, accept={}".format(-log_py_old, -log_py, accept_rate))
                # accept_rate = 1
                if np.random.uniform() > accept_rate:
                    beta[di_idx] = beta_idx_old
                    W[di, :] = Wdi_old
                    Phi[:, di_idx] = Phidi_old
                else:
                    log_py_old = log_py
                    log_lap_beta_old[di] = log_lap_beta
                    log_pbeta_old[di] = log_pbeta

        self.W_ = W
        self.beta_ = beta

    def _fit_loop_v1(self, x, y,
                     do_validation=False,
                     x_valid=None, y_valid=None,
                     callbacks=None, callback_metrics=None):
        N = x.shape[0]
        D = x.shape[1]
        dim_rf = self.dim_rf
        dprime = dim_rf * 2
        dprime_ext = dprime + 1
        scale_rf = 1.0 / np.sqrt(dim_rf)

        L = self.max_loop

        self.x_ = x
        self.y_ = y
        # tempt fix bug
        self.y_[self.y_ == 0] = -1

        kappa0 = self.kappa
        mu0 = np.zeros(D)
        nu0 = D + 3
        Psi0 = self.gamma * np.eye(D, D)

        K = int(dim_rf / 10)  # as in original code
        Z = np.zeros(dim_rf).astype(int)
        W = np.zeros((dim_rf, D))

        mu_lst = []
        cov_lst = []
        for k in range(K):
            mu_lst.append(np.zeros(D))
            cov_lst.append(np.zeros((D, D)))
        nk = np.zeros(K)
        for di in range(dim_rf):
            nk[Z[di]] += 1
        for k in range(K):
            mu_lst[k], cov_lst[k] = BANK.get_prior_giw(D, W[Z == k, :], kappa0, mu0, nu0, Psi0)

        for di in range(dim_rf):
            W[di, :] = np.random.multivariate_normal(mu_lst[Z[di]], cov_lst[Z[di]])

        beta = np.zeros(dprime_ext)
        # for di in range(dim_rf):
        #     di_idx = [di, di + dim_rf]
        #     n_di_idx = 2
        #     if di == dim_rf - 1:
        #         di_idx = [di, di + dim_rf, dprime]
        #         n_di_idx = 2
        #     beta[di_idx] = stats.multivariate_normal.rvs(np.zeros(n_di_idx), 1.0 / self.lbd)
        Wx = np.dot(W, self.x_.T).T

        Phi = np.ones((N, dprime_ext))
        Phi[:, 0:dim_rf] = np.cos(Wx)
        Phi[:, dim_rf:dprime] = np.sin(Wx)
        Phi *= scale_rf

        log_lap_beta_old = np.zeros(dim_rf)
        log_pbeta_old = np.zeros(dim_rf)

        for l in range(L):
            # print("W={}".format(W))
            # sample mu, cov
            for k in range(K):
                mu_lst[k], cov_lst[k] = BANK.get_prior_giw(D, W[Z == k, :], kappa0, mu0, nu0, Psi0)
                if nk[k] > 0:
                    print("mu = {}".format(mu_lst[k]))
                    print("cov = {}".format(cov_lst[k]))

            # sample Z
            for di in range(dim_rf):
                zdi = Z[di]
                # remove
                nk[zdi] -= 1

                pp = np.zeros(K)
                for k in range(K):
                    pp[k] = stats.multivariate_normal. \
                        logpdf(W[di, :], mu_lst[k], cov_lst[k])
                    if pp[k] < -300:
                        print(pp[k])
                pp -= np.max(pp)

                for k in range(K):
                    if nk[k] > 0:
                        pp[k] += np.log(nk[k])
                    else:
                        pp[k] += np.log(self.alpha)
                try:
                    pp = np.exp(pp - np.max(pp))
                    pp /= np.sum(pp)
                    zdi_new = np.argmax(np.random.multinomial(5, pp))
                except FloatingPointError:
                    print("Error pp={}".format(pp))
                    zdi_new = np.argmax(pp)

                Z[di] = zdi_new
                nk[zdi_new] += 1

            # sample W & beta
            exp_PhiXy = np.exp(np.sum(-Phi * beta, axis=1) * self.y_)  # (N,)
            log_py_old = -np.sum(np.log(1.0 + exp_PhiXy))
            print("loss={}".format(-log_py_old))

            for di in range(dim_rf):
                di_idx = [di, di + dim_rf]
                n_di_idx = 2
                if di == dim_rf - 1:
                    di_idx = [di, di + dim_rf, dprime]
                    n_di_idx = 3
                log_pbeta_old[di] = stats.multivariate_normal.logpdf(beta[di_idx], np.zeros(n_di_idx), 1.0 / self.lbd)
                lap_matrix_a = self.lbd * np.eye(n_di_idx)  # np.zeros((2, 2))
                for n in range(N):
                    phi_pair = Phi[n:1, di_idx]
                    lap_matrix_a += np.dot(phi_pair.T, phi_pair) * (exp_PhiXy[n] / (exp_PhiXy[n]) ** 2)
                log_lap_beta_old[di] = (0.5 * N * np.log(2 * np.pi)) \
                                       + (- 0.5 * np.log(np.abs(np.linalg.det(lap_matrix_a))))

            for di in range(dim_rf):
                di_idx = [di, di + dim_rf]
                n_di_idx = 2
                if di == dim_rf - 1:
                    di_idx = [di, di + dim_rf, dprime]
                    n_di_idx = 3

                zdi = Z[di]
                Wdi_old = W[di, :]
                Phidi_old = Phi[:, di_idx]
                W[di, :] = np.random.multivariate_normal(mu_lst[zdi], cov_lst[zdi])
                Wx_di_tmp = np.dot(self.x_, W[di, :].T)  # (N, 1)
                Phi[:, di] = np.cos(Wx_di_tmp) * scale_rf
                Phi[:, di + dim_rf] = np.sin(Wx_di_tmp) * scale_rf

                T = self.inner_epoch * N
                beta_idx_old = beta[di_idx]
                for t in range(T):
                    idx = np.random.randint(0, N, self.batch_size)  # (t,)
                    PhiX_t = np.sum(Phi[idx, :] * beta, axis=1)  # (t,)
                    PhiXy_t = PhiX_t * self.y_[idx]  # (t,)
                    exp_minus_PhiXy_t = np.exp(-PhiXy_t)
                    grad = self.lbd - self.y_[idx] * exp_minus_PhiXy_t / (exp_minus_PhiXy_t + 1)
                    # beta[di_idx] *= (1.0 * t) / (t + 1)
                    beta[di_idx] -= \
                        (1.0 / (self.lbd * (t + 1))) * np.sum(Phi[idx, :][:, di_idx].T * grad, axis=1) / self.batch_size

                exp_PhiXy = np.exp(np.sum(-Phi * beta, axis=1) * self.y_)
                log_py = -np.sum(np.log(1.0 + exp_PhiXy))
                try:
                    log_pbeta = stats.multivariate_normal.logpdf(beta[di_idx], np.zeros(n_di_idx), 1.0 / self.lbd)
                except FloatingPointError:
                    print("Error log_pbeta = {}".format(beta[di_idx]))
                lap_matrix_a = self.lbd * np.eye(n_di_idx)  # np.zeros((2, 2))
                for n in range(N):
                    phi_pair = Phi[n:1, di_idx]
                    lap_matrix_a += np.dot(phi_pair.T, phi_pair) * (exp_PhiXy[n] / (exp_PhiXy[n]) ** 2)
                log_lap_beta = \
                    (0.5 * N * np.log(2 * np.pi)) \
                    + (- 0.5 * np.log(np.abs(np.linalg.det(lap_matrix_a))))
                try:
                    accept_rate = min(1, np.exp((log_py + log_pbeta + log_lap_beta_old[di]) -
                                                (log_py_old + log_pbeta_old[di] + log_lap_beta)))
                except FloatingPointError:
                    print("Error:{},{},{} {},{},{}".format(log_py, log_pbeta, log_lap_beta_old[di],
                                                           log_py_old, log_pbeta_old[di], log_lap_beta))
                print("Accept 1:{},{},{} {},{},{}".format(log_py, log_pbeta, log_lap_beta_old[di],
                                                          log_py_old, log_pbeta_old[di], log_lap_beta))
                print("Loss old={}, new={}, accept={}".format(-log_py_old, -log_py, accept_rate))
                # accept_rate = 1
                if np.random.uniform() > accept_rate:
                    beta[di_idx] = beta_idx_old
                    W[di, :] = Wdi_old
                    Phi[:, di_idx] = Phidi_old
                else:
                    log_py_old = log_py
                    log_lap_beta_old[di] = log_lap_beta
                    log_pbeta_old[di] = log_pbeta

        self.W_ = W
        self.beta_ = beta

    def predict(self, x):
        dim_rf = self.dim_rf
        dprime = dim_rf * 2
        dprime_ext = dprime + 1
        scale_rf = 1.0 / np.sqrt(dim_rf)

        N_test = x.shape[0]
        y = np.ones(N_test)

        Wx = np.dot(self.W_, x.T).T
        Phi = np.ones((N_test, dprime_ext))
        Phi[:, 0:dim_rf] = np.cos(Wx)
        Phi[:, dim_rf:dprime] = np.sin(Wx)
        Phi *= scale_rf

        y[(np.sum(Phi * self.beta_, axis=1)) < 0] = 0
        return y

    def get_params(self, deep=True):
        out = super(BANK, self).get_params(deep=deep)
        out.update({
            'gamma': self.gamma,
            'kappa': self.kappa,
            'lbd': self.lbd,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(BANK, self).get_all_params(deep=deep)
        out.update({
            'W_': copy.deepcopy(self.W_),
            'beta_': copy.deepcopy(self.beta_),
        })
        return out
