from .char_rnn import CharRNN

try:
    from .gan import GAN
    from .dcgan import DCGAN
    from .gan1d import GAN1D
    from .gan2d import GAN2D
except ImportError:
    GAN = None
    DCGAN = None
    GAN1D = None
    GAN2D = None
