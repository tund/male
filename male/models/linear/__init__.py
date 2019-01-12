try:
    from .glm import GLM
except ImportError as e:
    print('[WARNING]', e)

try:
    from .tensorflow_glm import TensorFlowGLM
except ImportError as e:
    print('[WARNING]', e)
