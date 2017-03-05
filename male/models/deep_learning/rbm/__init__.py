from .srbm import SupervisedRBM
from .bbrbm import BernoulliBernoulliRBM

try:
    from .tensorflow_bbrbm import BernoulliBernoulliTensorFlowRBM
except ImportError:
    BernoulliBernoulliTensorFlowRBM = None
