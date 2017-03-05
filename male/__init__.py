from __future__ import absolute_import

from os.path import expanduser

from . import model
from .model import Model

try:
    from .tensorflow_model import TensorFlowModel
except ImportError:
    TensorFlowModel = None

HOME = expanduser("~")

__version__ = '0.1.0'
