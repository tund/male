import os
import platform
from os.path import expanduser

import json
from builtins import str

from .common import epsilon
from .common import set_epsilon
from .common import random_seed
from .common import set_random_seed
from .common import matplotlib_backend
from .common import set_matplotlib_backend
from .common import data_dir
from .common import set_data_dir
from .common import model_dir
from .common import set_model_dir
from .common import remote_data_dir
from .common import set_remote_data_dir
from .common import remote_model_dir
from .common import set_remote_model_dir

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
    _matplotlib_backend = _config.get('matplotlib_backend', matplotlib_backend())
    assert isinstance(_matplotlib_backend, str)
    _data_dir = _config.get('data_dir', data_dir())
    assert isinstance(_data_dir, str)
    _model_dir = _config.get('model_dir', model_dir())
    assert isinstance(_model_dir, str)
    _remote_data_dir = _config.get('remote_data_dir', remote_data_dir())
    assert isinstance(_remote_data_dir, str)
    _remote_model_dir = _config.get('remote_model_dir', remote_model_dir())
    assert isinstance(_remote_model_dir, str)

    set_epsilon(_epsilon)
    set_random_seed(_random_seed)
    set_matplotlib_backend(_matplotlib_backend)
    set_data_dir(_data_dir)
    set_model_dir(_model_dir)
    set_remote_data_dir(_remote_data_dir)
    set_remote_model_dir(_remote_model_dir)

# save config file
# if not os.path.exists(_config_path):
_config = {'epsilon': epsilon(),
           'random_seed': random_seed(),
           'matplotlib_backend': matplotlib_backend(),
           'data_dir': data_dir(),
           'model_dir': model_dir(),
           'remote_data_dir': remote_data_dir(),
           'remote_model_dir': remote_model_dir()}
with open(_config_path, 'w') as f:
    f.write(json.dumps(_config, indent=4))
    # </editor-fold>
