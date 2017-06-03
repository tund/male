from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male import Model
from male import model_dir
from male import random_seed
from male.datasets import demo
from male.optimizers import SGD
from male.models.linear import GLM
from male.callbacks import Display
from male.callbacks import ImageSaver
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_early_stopping():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    optz = SGD(learning_rate=0.01)
    clf = GLM(model_name="early_stopping_callback",
              link='softmax',
              loss='softmax',
              optimizer=optz,
              num_epochs=20,
              batch_size=10,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[early_stopping],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Model has been stopped at epoch #{0:d}".format(clf.epoch))
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Continue training...")
    clf.fit(x, y)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Model has been stopped at epoch #{0:d}".format(clf.epoch))
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Disable early stopping and continue training to the end...")
    clf.callbacks = []
    clf.fit(x, y)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_checkpoint():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    filepath = os.path.join(model_dir(), "male/glm/checkpoint_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    optz = SGD(learning_rate=0.01)
    clf = GLM(model_name="checkpoint_callback",
              link='softmax',
              loss='softmax',
              optimizer=optz,
              num_epochs=5,
              batch_size=10,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    model_filepath = filepath.format(epoch=5, val_loss=0.968786)
    print("Load model at checkpoint: ", model_filepath, ", and predict:")
    clf1 = Model.load_model(model_filepath)
    train_err = 1.0 - clf1.score(x_train, y_train)
    test_err = 1.0 - clf1.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_display_callbacks(block_figure_on_end=False):
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("Number of training samples = {}".format(x_train.shape[0]))

    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    err_display = Display(title="Error curves",
                          dpi='auto',
                          layout=(1, 1),
                          freq=1,
                          block_on_end=block_figure_on_end,
                          monitor=[{'metrics': ['err', 'val_err'],
                                    'type': 'line',
                                    'title': "Learning errors",
                                    'xlabel': "epoch",
                                    'ylabel': "error",
                                    }])
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(model_dir(), "male/callbacks/"
                                                               "display/loss/loss_{epoch:04d}.png"),
                                     os.path.join(model_dir(), "male/callbacks/"
                                                               "display/loss/loss_{epoch:04d}.pdf")],
                           monitor=[{'metrics': ['loss', 'val_loss'],
                                     'type': 'line',
                                     'labels': ["training loss", "validation loss"],
                                     'title': "Learning losses",
                                     'xlabel': "epoch",
                                     'xlabel_params': {'fontsize': 50},
                                     'ylabel': "loss",
                                     },
                                    {'metrics': ['err', 'val_err'],
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
                             block_on_end=block_figure_on_end,
                             filepath=os.path.join(model_dir(), "male/callbacks/display/"
                                                                "weights/weights_{epoch:04d}.png"),
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    optz = SGD(learning_rate=0.001)
    clf = GLM(model_name="display_callbacks",
              link='softmax',
              loss='softmax',
              optimizer=optz,
              num_epochs=20,
              batch_size=100,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[loss_display, weight_display, err_display],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)
    print("Training error = %.4f" % (1.0 - clf.score(x_train, y_train)))
    print("Testing error = %.4f" % (1.0 - clf.score(x_test, y_test)))


def test_image_saver_callback():
    np.random.seed(random_seed())

    (x_train, y_train), (_, _) = demo.load_mnist()
    (cifar10_train, _), (_, _) = demo.load_cifar10()

    imgsaver1 = ImageSaver(freq=1,
                           filepath=os.path.join(model_dir(), "male/callbacks/imagesaver/"
                                                              "mnist/mnist_{epoch:04d}.png"),
                           monitor={'metrics': 'x_data',
                                    'img_size': (28, 28, 1),
                                    'tile_shape': (10, 10),
                                    'images': x_train[:100].reshape([-1, 28, 28, 1])})
    imgsaver2 = ImageSaver(freq=1,
                           filepath=os.path.join(model_dir(), "male/callbacks/imagesaver/"
                                                              "cifar10/cifar10_{epoch:04d}.png"),
                           monitor={'metrics': 'x_data',
                                    'img_size': (32, 32, 3),
                                    'tile_shape': (10, 10),
                                    'images': cifar10_train[:100].reshape([-1, 32, 32, 3])})

    optz = SGD(learning_rate=0.001)
    clf = GLM(model_name="imagesaver_callback",
              link='softmax',
              loss='softmax',
              optimizer=optz,
              num_epochs=4,
              batch_size=100,
              task='classification',
              callbacks=[imgsaver1, imgsaver2],
              random_state=random_seed(),
              verbose=1)
    clf.fit(x_train, y_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_early_stopping()
    # test_checkpoint()
    # test_display_callbacks(block_figure_on_end=True)
    # test_image_saver_callback()
