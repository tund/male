try:
    from .gan import GAN
    from .dcgan import DCGAN
    from .weighted_gan import WeightedGAN
    from .dfm import DFM
    from .gank import GANK
    from .m2gan import M2GAN
    from .gan1d import GAN1D
    from .gan2d import GAN2D
    from .grn1d import GRN1D
    from .d2gan1d import D2GAN1D
    from .d2gan2d import D2GAN2D
    from .mggan2d import MGGAN2D
except ImportError:
    GAN = None
    DCGAN = None
    WeightedGAN = None
    DFM = None
    GANK = None
    M2GAN = None
    GAN1D = None
    GAN2D = None
    GRN1D = None
    D2GAN1D = None
    D2GAN2D = None
    MGGAN2D = None
