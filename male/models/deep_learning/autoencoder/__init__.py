try:
    from .keras_vae import KerasVAE
except ImportError as e:
    print('[WARNING]', e)
