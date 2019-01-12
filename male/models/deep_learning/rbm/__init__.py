try:
    from .bbrbm import BernoulliBernoulliRBM

except ImportError as e:
    print('[WARNING]', e)

try:
    from .nrbm import NonnegativeRBM
except ImportError as e:
    print('[WARNING]', e)

try:
    from .srbm import SupervisedRBM
except ImportError as e:
    print('[WARNING]', e)

try:
    from .ssrbm import SemiSupervisedRBM
except ImportError as e:
    print('[WARNING]', e)

try:
    from .efrbm import EFRBM
except ImportError as e:
    print('[WARNING]', e)

try:
    from .rsrbm import ReplicatedSoftmaxRBM
except ImportError as e:
    print('[WARNING]', e)

try:
    from .tensorflow_bbrbm import BernoulliBernoulliTensorFlowRBM
except ImportError as e:
    print('[WARNING]', e)
