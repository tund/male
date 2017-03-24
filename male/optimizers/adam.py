from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .. import Optimizer


class Adam(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        kwargs['learning_rate'] = learning_rate
        super(Adam, self).__init__(**kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay

    def init_params(self, **kwargs):
        super(Adam, self).init_params(**kwargs)
        self.iter_ = 0
        self.initial_decay_ = self.decay
        self.first_moments_ = [np.zeros(p.shape) for p in self.params_]
        self.second_moments_ = [np.zeros(p.shape) for p in self.params_]

    def update_params(self, *args, **kwargs):
        grads = self.grad_func_(*args)

        lr = self.learning_rate
        if self.initial_decay_ > 0:
            lr *= (1. / (1. + self.decay * self.iter_))

        self.iter_ += 1
        lr_t = lr * (np.sqrt(1. - self.beta_2 ** self.iter_) / (1. - self.beta_1 ** self.iter_))

        for p, g, m, v in zip(self.params_, grads, self.first_moments_, self.second_moments_):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * g * g
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)

            m[:] = m_t
            v[:] = v_t
            p[:] = p_t


class Adamax(Adam):
    """Adamax optimizer from Adam paper's Section 7.

    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, learning_rate=0.002, **kwargs):
        kwargs['learning_rate'] = learning_rate
        super(Adamax, self).__init__(**kwargs)

    def update_params(self, *args, **kwargs):
        grads = self.grad_func_(*args)

        lr = self.learning_rate
        if self.initial_decay_ > 0:
            lr *= (1. / (1. + self.decay * self.iter_))

        self.iter_ += 1
        lr_t = lr / (1. - self.beta_1 ** self.iter_)

        # Note that: self.second_moments_ is now exponentially weighted infinity norm
        for p, g, m, u in zip(self.params_, grads, self.first_moments_, self.second_moments_):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = np.maximum(self.beta_2 * u, np.abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)

            m[:] = m_t
            u[:] = u_t
            p[:] = p_t


class Nadam(Adam):
    """Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, learning_rate=0.002, schedule_decay=0.004, **kwargs):
        kwargs['learning_rate'] = learning_rate
        super(Nadam, self).__init__(**kwargs)
        self.schedule_decay = schedule_decay

    def init_params(self, **kwargs):
        super(Nadam, self).init_params(**kwargs)
        self.m_schedule_ = 1.0

    def update_params(self, *args, **kwargs):
        grads = self.grad_func_(*args)

        self.iter_ += 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (np.power(0.96, self.iter_ * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (np.power(0.96, (self.iter_ + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule_ * momentum_cache_t
        m_schedule_next = self.m_schedule_ * momentum_cache_t * momentum_cache_t_1
        self.m_schedule_ = m_schedule_new

        for p, g, m, v in zip(self.params_, grads, self.first_moments_, self.second_moments_):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * g * g
            v_t_prime = v_t / (1. - self.beta_2 ** self.iter_)
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            m[:] = m_t
            v[:] = v_t

            p_t = p - self.learning_rate * m_t_bar / (np.sqrt(v_t_prime) + self.epsilon)
            p[:] = p_t
