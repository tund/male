from __future__ import absolute_import

import os
import platform
from os.path import expanduser
import json
from builtins import str

from . import model
from .model import Model
from .optimizer import Optimizer
from .common import epsilon
from .common import set_epsilon
from .common import random_seed
from .common import set_random_seed
from .common import data_dir
from .common import set_data_dir
from .common import model_dir
from .common import set_model_dir
from .common import remote_data_dir
from .common import set_remote_data_dir
from .common import remote_model_dir
from .common import set_remote_model_dir

try:
    from .tensorflow_model import TensorFlowModel
except ImportError:
    TensorFlowModel = None

HOME = expanduser("~")

# <editor-fold desc="Directory">
_male_base_dir = HOME
if not os.access(_male_base_dir, os.W_OK):
    if platform.system() == "Windows":
        _male_base_dir = "C:/tmp"
    elif platform.system() == "Linux":
        _male_base_dir = "/tmp"
    elif platform.system() == "Darwin":
        _male_base_dir = "/tmp"
    else:
        raise NameError("Cannot recognize platform.")

_male_dir = os.path.join(_male_base_dir, '.male')
if not os.path.exists(_male_dir):
    os.makedirs(_male_dir)
# </editor-fold>

# <editor-fold desc="Configuration">
_config_path = os.path.expanduser(os.path.join(_male_dir, 'male.json'))
if os.path.exists(_config_path):
    _config = json.load(open(_config_path))
    _epsilon = _config.get('epsilon', epsilon())
    assert isinstance(_epsilon, float)
    _random_seed = _config.get('random_seed', random_seed())
    assert isinstance(_random_seed, int)
    _data_dir = _config.get('data_dir', os.path.join(_male_dir, 'datasets'))
    assert isinstance(_data_dir, str)
    _model_dir = _config.get('model_dir', os.path.join(_male_dir, 'models'))
    assert isinstance(_model_dir, str)
    _remote_data_dir = _config.get('remote_data_dir', 'http://prada-research.net/demo/datasets')
    assert isinstance(_remote_data_dir, str)
    _remote_model_dir = _config.get('remote_model_dir', 'http://prada-research.net/demo/models')
    assert isinstance(_remote_model_dir, str)

    set_epsilon(_epsilon)
    set_random_seed(_random_seed)
    set_data_dir(_data_dir)
    set_model_dir(_model_dir)
    set_remote_data_dir(_remote_data_dir)
    set_remote_model_dir(_remote_model_dir)

# save config file
if not os.path.exists(_config_path):
    _config = {'epsilon': epsilon(),
               'random_seed': random_seed(),
               'data_dir': os.path.join(_male_dir, 'datasets'),
               'model_dir': os.path.join(_male_dir, 'models'),
               'remote_data_dir': 'http://prada-research.net/demo/datasets',
               'remote_model_dir': 'http://prada-research.net/demo/models'}
    with open(_config_path, 'w') as f:
        f.write(json.dumps(_config, indent=4))
# </editor-fold>

__version__ = '0.1.0'
