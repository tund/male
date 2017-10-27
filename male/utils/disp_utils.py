from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import matplotlib
import numpy as np
from matplotlib.colors import ColorConverter

from ..configs import matplotlib_backend

# if matplotlib_backend() != "default":
#     matplotlib.use(matplotlib_backend())
import matplotlib.pyplot as plt


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not
    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='float32')

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt) + X.min()

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def create_image_grid(x, img_size=None, tile_shape=None, output_pixel_vals=False, **kwargs):
    if tile_shape is None:
        tile_shape = (x.shape[0], 1)
    if img_size[2] == 1:
        img = tile_raster_images(x.reshape([x.shape[0], -1]),
                                 img_shape=(img_size[0], img_size[1]),
                                 tile_shape=tile_shape,
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=output_pixel_vals)
    else:
        img = tile_raster_images((x[..., 0], x[..., 1], x[..., 2], None),
                                 img_shape=(img_size[0], img_size[1]),
                                 tile_shape=tile_shape,
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=output_pixel_vals)
    return img


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def get_screen_resolution():
    import matplotlib.pyplot as plt
    plt.figure()
    mgr = plt.get_current_fig_manager()
    mgr.full_screen_toggle()  # primitive but works to get screen size
    py = mgr.canvas.height()
    px = mgr.canvas.width()
    plt.close()
    return px, py


def get_figure_dpi():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    dpi = fig.get_dpi()
    plt.close()
    return dpi


def visualize_classification_prediction(estimator, x_train, y_train,
                                        show=False, block_on_end=False, **kwargs):
    if x_train.shape[1] > 2:
        print('Support only 2D datasets')
        return
    if len(np.unique(y_train)) > 5:
        print('Not support for n_classes > 5')
        return
    grid_size = 500 if 'grid_size' not in kwargs else kwargs['grid_size']
    marker_size = 500 if 'marker_size' not in kwargs else kwargs['marker_size']
    top = None if 'top' not in kwargs else kwargs['top']
    bottom = None if 'bottom' not in kwargs else kwargs['bottom']
    left = None if 'left' not in kwargs else kwargs['left']
    right = None if 'right' not in kwargs else kwargs['right']

    epoch = kwargs['epoch']

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
    x_scale = (right - left) / grid_size
    y_scale = (top - bottom) / grid_size

    print('Drawing at epoch {}...'.format(epoch))
    n_test = x_test.shape[0]
    y_test = estimator.predict(x_test)
    y_test_zidx = y_test.astype(int).copy()
    y_train_zidx = estimator.label_encoder.inverse_transform(y_train.astype(int))

    img = np.zeros((n_test, 3))
    n_classes = estimator.num_classes
    bg_colors_hex = np.array(["#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6"])  # Bugs
    fore_colors_hex = np.array(["#1F78B4", "#33A02C", "#E31A1C", "#FF7F00", "#6A3D9A"])  # Bugs
    converter = ColorConverter()

    bg_colors = np.zeros((len(bg_colors_hex), 3))
    for ci in range(len(fore_colors_hex)):
        bg_colors[ci, :] = converter.to_rgb(bg_colors_hex[ci])

    for ci in range(n_classes):
        img[y_test_zidx == ci, :] = bg_colors[ci]

    img = img.reshape((grid_size, grid_size, 3))

    plt.imshow(img)
    plt.scatter((x_train[:, 0] - left) / x_scale, (x_train[:, 1] - bottom) / y_scale,
                s=marker_size, color=fore_colors_hex[y_train_zidx.astype(int)])

    axes = plt.gca()
    axes.set_xlim([0, grid_size])
    axes.set_ylim([0, grid_size])
    # plt.legend()
    kwargs['ax'] = axes
    if show:
        plt.show(block=block_on_end)
