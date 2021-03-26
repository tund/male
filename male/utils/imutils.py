from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import numpy as np
from PIL import ImageColor


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


def resize(img, new_size, keep_aspect=True, crop=True, padcolor='white'):
    """Resize an image to a target size with some particular options.

    # Arguments:
        img: the input image in Numpy array format.
        new_size: the target size `(target_height, target_width)`.
        keep_aspect: whether to keep the aspect ratio between height and width, default=True.
        crop: if True: rescale the image so that:
            (new_height = target_height and new_width >= target_width, and then crop the width)
            or vice versa;
            if False: rescale the image so that:
            (new_height = target_height and new_width <= target_width, and then pad the width)
            or vice versa.
        padcolor: the background color when padding.
    # Returns:
        The resized image.
    """
    new_h, new_w = int(new_size[0]), int(new_size[1])

    if (new_h, new_w) != img.shape[:2]:
        if not keep_aspect:
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            if not crop:
                # Can consider using image.thumbnail(new_size, Image.ANTIALIAS) when
                # both dimensions of image are equal or greater than those of the new image.
                r = max(img.shape[0] / new_h, img.shape[1] / new_w)
                im_rz = cv2.resize(img, (int(img.shape[1] / r), int(img.shape[0] / r)), interpolation=cv2.INTER_AREA)
                if padcolor is None:
                    new_img = im_rz
                else:
                    margin_h = (new_h - im_rz.shape[0]) // 2
                    margin_w = (new_w - im_rz.shape[1]) // 2
                    rgb = ImageColor.getrgb(padcolor)
                    new_img = np.stack([np.ones([new_h, new_w], img.dtype) * rgb[0],
                                        np.ones([new_h, new_w], img.dtype) * rgb[1],
                                        np.ones([new_h, new_w], img.dtype) * rgb[2]], axis=2)
                    new_img[margin_h:margin_h + im_rz.shape[0], margin_w:margin_w + im_rz.shape[1], :] = im_rz
            else:
                r = min(img.shape[0] / new_h, img.shape[1] / new_w)
                im_rz = cv2.resize(img, (int(np.ceil(img.shape[1] / r)), int(np.ceil(img.shape[0] / r))),
                                   interpolation=cv2.INTER_AREA)
                top = (im_rz.shape[0] - new_h) // 2
                left = (im_rz.shape[1] - new_w) // 2
                bottom = top + new_h
                right = left + new_w
                new_img = im_rz[top:bottom, left:right, :]

    else:
        new_img = img

    return new_img
