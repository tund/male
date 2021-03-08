from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2


def load_image_file(file, mode='RGB'):
    """Loads an image file (.jpg, .png, etc) into a numpy array, and convert color mode if needed.

    # Arguments:
        file: image file name or file object to load.
        mode: color mode ('RGB', 'L', ...).

    # Returns:
        An image in numpy array format.
    """
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif mode == 'L':
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img
