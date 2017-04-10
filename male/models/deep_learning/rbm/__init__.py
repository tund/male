from .efrbm import EFRBM
from .nrbm import NonnegativeRBM
from .srbm import SupervisedRBM
from .ssrbm import SemiSupervisedRBM
from .bbrbm import BernoulliBernoulliRBM

try:
    from .tensorflow_bbrbm import BernoulliBernoulliTensorFlowRBM
except ImportError:
    BernoulliBernoulliTensorFlowRBM = None
