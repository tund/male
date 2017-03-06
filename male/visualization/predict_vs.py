from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter

from male.model import Model


def visualize_classification_prediction(estimator, x_train, y_train,
                                        grid_size=500, marker_size = 10,
                                        left=None, right=None, top=None, bottom=None):
    if x_train.shape[1] > 2:
        print('Support only 2D datasets')
        return
    if len(np.unique(y_train)) > 5:
        print('Not support for n_classes > 5')
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
    x_scale = (right-left) / grid_size
    y_scale = (top - bottom) / grid_size
    # print(left,right)
    # print(top, bottom)
    # print(x_scale)
    # print(y_scale)
    # print(x_test.shape)

    print('Drawing ...')
    n_test = x_test.shape[0]
    y_test = estimator.predict(x_test)
    y_test_zidx = estimator.label_encoder_.inverse_transform(y_test.astype(int))
    y_train_zidx = estimator.label_encoder_.inverse_transform(y_train.astype(int))

    img = np.zeros((n_test, 3))
    n_classes = estimator.num_classes_
    bg_colors_hex = np.array(["#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6"]) # Bugs
    fore_colors_hex = np.array(["#1F78B4", "#33A02C", "#E31A1C",  "#FF7F00", "#6A3D9A"])  # Bugs
    converter = ColorConverter()

    bg_colors = np.zeros((len(bg_colors_hex), 3))
    for ci in xrange(len(fore_colors_hex)):
        bg_colors[ci, :] = converter.to_rgb(bg_colors_hex[ci])

    for ci in xrange(n_classes):
        img[y_test_zidx == ci, :] = bg_colors[ci]

    img = img.reshape((grid_size, grid_size, 3))

    plt.imshow(img)
    plt.scatter((x_train[:, 0] - left) / x_scale, (x_train[:, 1] - bottom)/ y_scale,
                s=marker_size, color=fore_colors_hex[y_train_zidx.astype(int)])

    axes = plt.gca()
    axes.set_xlim([0, grid_size])
    axes.set_ylim([0, grid_size])
    plt.show()






