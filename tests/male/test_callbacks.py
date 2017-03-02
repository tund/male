from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male import GLM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_display_callbacks():
    from male import HOME

    x_train = np.random.randn(10000, 10)
    y_train = np.random.randint(0, 10, size=(10000))
    x_test = np.random.randn(1000, 10)
    y_test = np.random.randint(0, 10, size=(1000))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(HOME, "rmodel/male/glm/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
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
                             freq=10,
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    clf = GLM(model_name="mnist_glm_softmax",
              link='softmax',
              loss='softmax',
              optimizer='sgd',
              num_epochs=100,
              batch_size=100,
              learning_rate=0.001,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint, loss_display, weight_display],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=6789,
              verbose=1)

    clf.fit(x, y)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    # pytest.main([__file__])
    test_display_callbacks()
