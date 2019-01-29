try:
    from .char_rnn import CharRNN
except ImportError as e:
    CharRNN = None
    print('[WARNING]', e)

try:
    from .gan import GAN
except ImportError as e:
    GAN = None
    print('[WARNING]', e)

try:
    from .dcgan import DCGAN
except ImportError as e:
    DCGAN = None
    print('[WARNING]', e)

try:
    from .gan1d import GAN1D
except ImportError as e:
    GAN1D = None
    print('[WARNING]', e)

try:
    from .gan2d import GAN2D
except ImportError as e:
    GAN2D = None
    print('[WARNING]', e)

try:
    from .d2gan1d import D2GAN1D
except ImportError as e:
    print('[WARNING]', e)

try:
    from .d2gan2d import D2GAN2D
except ImportError as e:
    D2GAN2D = None
    print('[WARNING]', e)

try:
    from .dfm import DFM
except ImportError as e:
    DFM = None
    print('[WARNING]', e)

try:
    from .gank import GANK
except ImportError as e:
    GANK = None
    print('[WARNING]', e)

try:
    from .grn1d import GRN1D
except ImportError as e:
    GRN1D = None
    print('[WARNING]', e)

try:
    from .m2gan import M2GAN
except ImportError as e:
    M2GAN = None
    print('[WARNING]', e)

try:
    from .mggan2d import MGGAN2D
except ImportError as e:
    MGGAN2D = None
    print('[WARNING]', e)

try:
    from .weighted_gan import WeightedGAN
except ImportError as e:
    WeightedGAN = None
    print('[WARNING]', e)

try:
    from .cgan import CGAN
except ImportError as e:
    CGAN = None
    print('[WARNING]', e)

try:
    from .cgan import CGANv1
except ImportError as e:
    CGANv1 = None
    print('[WARNING]', e)

try:
    from .wgan import WGAN
except ImportError as e:
    WGAN = None
    print('[WARNING]', e)

try:
    from .wgan_gp import WGAN_GP
except ImportError as e:
    WGAN_GP = None
    print('[WARNING]', e)

try:
    from .wgan_gp_resnet import WGAN_GP_ResNet
except ImportError as e:
    WGAN_GP_ResNet = None
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
