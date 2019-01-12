try:
    from .mlp import MLP
except ImportError as e:
    print('[WARNING]', e)

try:
    from .tensorflow_nets import SoftmaxClassifier
except ImportError as e:
    print('[WARNING]', e)

try:
    from .tensorflow_nets import MLProjection
except ImportError as e:
    print('[WARNING]', e)

'''
# This way looks nice, but we cannot browse code (Go to definition, declaration, ...)
import_models = [('from .adsf import ASDF', 'ASDF = None'),
                 ('from .qwer import QWER', 'QWER = None')]
for import_cmd, assign_cmd in import_models:
    try:
        exec(import_cmd)
    except Exception as e:
        exec(assign_cmd)
        print('[WARNING]', e)
'''
