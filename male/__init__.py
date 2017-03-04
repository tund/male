from __future__ import absolute_import

from os.path import expanduser

from . import model
from .model import Model
from .tensorflow_model import TensorFlowModel

HOME = expanduser("~")

__version__ = '0.1.0'
