from __future__ import absolute_import

from . import configs
from .model import Model
from .optimizer import Optimizer

try:
    from .tensorflow_model import TensorFlowModel
except ImportError:
    TensorFlowModel = None

__version__ = '0.1.0'
