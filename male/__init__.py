from __future__ import absolute_import

from . import model
from .model import Model
from .models.glm import GLM

from os.path import expanduser

HOME = expanduser("~")

__version__ = '0.1.0'
