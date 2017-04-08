try:
    from .gan import GAN
    from .gan1d import GAN1D
    from .grn1d import GRN1D
except ImportError:
    GAN = None
    GAN1D = None
    GRN1D = None
