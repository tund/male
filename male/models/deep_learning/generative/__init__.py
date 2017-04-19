try:
    from .gan import GAN
    from .gan1d import GAN1D
    from .gan2d import GAN2D
    from .grn1d import GRN1D
except ImportError:
    GAN = None
    GAN1D = None
    GAN2D = None
    GRN1D = None
