from .glm import GLM

try:
    from .tensorflow_glm import TensorFlowGLM
except ImportError:
    TensorFlowGLM = None
