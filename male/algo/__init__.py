try:
    from .wasserstein import emd
except ImportError as e:
    emd = None
    print('[WARNING]', e)
