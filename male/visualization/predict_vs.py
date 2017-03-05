from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from male.model import Model


def visualize_classification_prediction(estimator, x_train, y_train,
                                        grid_size=500, left=None, right=None, top=None, bottom=None):
    if x_train.shape[1] > 2:
        print('Support only 2D datasets\n')
        return

    # fit training set to model
    print('Training ...\n')
    estimator.fit(x_train, y_train)
    print('Finished training\n')

    if left is None:
        left = np.min(x_train, axis=0)[0]
    if right is None:
        right = np.max(x_train, axis=0)[0]
    if top is None:
        top = np.max(x_train, axis=0)[1]
    if bottom is None:
        bottom = np.min(x_train, axis=0)[1]

    x_axis = np.linspace(left, right, grid_size)
    y_axis = np.linspace(bottom, top, grid_size)
    x_axis_v, y_axis_v = np.meshgrid(x_axis, y_axis)
    x_test = np.vstack((x_axis_v.ravel(), y_axis_v.ravel()))
    print(x_test.shape)

    n_test = x_test.shape[0]
    y_test = estimator.predict(x_test)
    y_test_zidx = estimator.label_encoder_.inverse_transform(y_test)
    y_train_zidx = estimator.label_encoder_.inverse_transform(y_train)

    img = np.zeros((n_test, 3))
    n_classes = estimator.num_classes_
    colors = None  # Bugs
    for ci in xrange(n_classes):
        img[y_test_zidx == ci, :] = colors[ci]

    img = img.reshape((grid_size, grid_size, 3))

    plt.imshow(img)





