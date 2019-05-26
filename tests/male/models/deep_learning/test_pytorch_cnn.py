from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male.datasets import demo
from male.configs import model_dir
from male.configs import random_seed
from male.callbacks import Display

from male import PyTorchModel
from male.models.deep_learning import PyTorchConvNet

import matplotlib.pyplot as plt


def test_pytorch_convnet_v1(show=False, block_figure_on_end=False):
    print("========== Test PytorchConvNetv1 ==========")

    np.random.seed(random_seed())

    err_display = Display(title="Error curves",
                          dpi='auto',
                          layout=(1, 1),
                          freq=1,
                          show=show,
                          block_on_end=block_figure_on_end,
                          monitor=[{'metrics': ['err'],
                                    'type': 'line',
                                    'title': "Learning errors",
                                    'xlabel': "epoch",
                                    'ylabel': "error",
                                    }])
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
                           show=show,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(model_dir(), "male/PyTorchConvNet/"
                                                               "loss/loss_{epoch:04d}.png"),
                                     os.path.join(model_dir(), "male/PyTorchConvNet/"
                                                               "loss/loss_{epoch:04d}.pdf")],
                           monitor=[{'metrics': ['loss'],
                                     'type': 'line',
                                     'labels': ["training loss"],
                                     'title': "Learning losses",
                                     'xlabel': "epoch",
                                     'xlabel_params': {'fontsize': 50},
                                     'ylabel': "loss",
                                     },
                                    {'metrics': ['err'],
                                     'type': 'line',
                                     'title': "Learning errors",
                                     'xlabel': "epoch",
                                     'ylabel': "error",
                                     },
                                    {'metrics': ['err'],
                                     'type': 'line',
                                     'labels': ["training error"],
                                     'title': "Learning errors",
                                     'xlabel': "epoch",
                                     'ylabel': "error",
                                     },
                                    ])

    weight_display = Display(title="Filters",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(6, 15),
                             freq=1,
                             show=show,
                             block_on_end=block_figure_on_end,
                             filepath=os.path.join(model_dir(), "male/PyTorchConvNet/"
                                                                "weights/weights_{epoch:04d}.png"),
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    # Construct the model.
    clf = PyTorchConvNet(model_name='PyTorchConvNet',
                         num_epochs=1,
                         batch_size=64,
                         metrics=['loss', 'err'],
                         callbacks=[loss_display, err_display],
                         random_state=random_seed(),
                         verbose=1)

    print('Show some of the training images.')
    if False:
        clf.num_epochs = 0
        # Build the network as well as data loaders.
        clf.fit()
        import torchvision

        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show(block=True)

        # get some random training images
        dataiter = iter(clf.trainloader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % clf.classes[labels[j]] for j in range(4)))
        clf.num_epochs = 1

    clf.fit()
    print('Testing accuracy = %.4f' % (100 * clf.acc_test()))

    save_file_path = clf.save()
    clf = PyTorchModel.load_model(save_file_path)
    clf.num_epochs = 4
    clf.fit()
    print('Testing accuracy = %.4f' % (100 * clf.acc_test()))


if __name__ == '__main__':
    # pytest.main([__file__])
    test_pytorch_convnet_v1(show=True, block_figure_on_end=True)
