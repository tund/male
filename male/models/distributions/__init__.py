import abc
from sklearn.utils import check_random_state


class Distribution(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.random_engine = check_random_state(self.random_state)

    def set_random_engine(self, random_engine):
        self.random_engine = random_engine

    def sample(self, num_samples, **kwargs):
        pass

    def logpdf(self, samples):
        pass


from .mixture import GMM
from .mixture import GMM1D
from .uniform import Uniform
from .uniform import Uniform1D
from .gaussian import Gaussian
from .gaussian import Gaussian1D
from .gaussian import InverseGaussian1D
