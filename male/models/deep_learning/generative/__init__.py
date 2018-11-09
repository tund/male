from .char_rnn import CharRNN

try:
    from .gan import GAN
    from .dcgan import DCGAN
    from .gan1d import GAN1D
    from .gan2d import GAN2D
    from .cgan import CGAN
    from .cgan import CGANv1
    from .wgan import WGAN
    from .wgan_gp import WGAN_GP
except ImportError:
    GAN = None
    DCGAN = None
    GAN1D = None
    GAN2D = None
    CGAN = None
    CGANv1 = None
    WGAN = None
    WGAN_GP = None
