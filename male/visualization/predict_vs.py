from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

from male.model import Model


def visualize_classification_prediction(estimator, x_train, y_train,
                                        grid_size=500, left=None, right=None, top=None, bottom=None):
    if x_train.shape[1] > 2:
        print('Support only 2D datasets')
        return

    # fit training set to model
    print('Training ...')
    estimator.fit(x_train, y_train)
    print('Finished training')

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
    x_test = x_test.T
    print(x_test.shape)

    n_test = x_test.shape[0]
    y_test = estimator.predict(x_test)
    y_test_zidx = estimator.label_encoder_.inverse_transform(y_test.astype(int))
    y_train_zidx = estimator.label_encoder_.inverse_transform(y_train.astype(int))

    img = np.zeros((n_test, 3))
    n_classes = estimator.num_classes_
    bg_colors_hex = np.array(["#00FFFF", "#00FFFF", "#FFFF66"]) # Bugs
    fore_colors_hex = ["#0000FF", "#00FF00", "#FF0066"]  # Bugs
    converter = ColorConverter()

    fore_colors = np.zeros((len(fore_colors_hex), 3))
    for ci in xrange(len(fore_colors_hex)):
        fore_colors[ci, :] = converter.to_rgb(fore_colors_hex[ci])

    for ci in xrange(n_classes):
        img[y_test_zidx == ci, :] = fore_colors[ci]

    img = img.reshape((grid_size, grid_size, 3))

    plt.imshow(img)
    plt.scatter(x_train[:, 0], x_train[:,1], s=5, c=bg_colors_hex[y_train_zidx.astype(int)], alpha=0.5)

    plt.show()






