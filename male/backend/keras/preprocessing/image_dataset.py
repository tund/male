import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.preprocessing.image_dataset import path_to_image

AUTOTUNE = tf.data.experimental.AUTOTUNE


def image_dataset_from_dataframe(df,
                                 class_names,
                                 label_mode='int',
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 buffer_size=None,
                                 seed=None,
                                 interpolation='bilinear'):
    """Generates a `tf.data.Dataset` from image files in a pandas dataframe.
    Supported image formats: jpeg, png, bmp, gif.
    Animated gifs are truncated to the first frame.

    # Arguments:
        df: Pandas dataframe.
        class_names: a list or tuple of class names.
        label_mode:
            - 'int': means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - 'categorical' means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - 'binary' means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - None (no labels).
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                    Whether the images will be converted to
                    have 1, 3, or 4 channels.
        batch_size: Size of the batches of data. Default: 32.
        image_size: Size to resize images to after they are read from disk.
                    Defaults to `(256, 256)`.
                    Since the pipeline processes batches of images that must all have
                    the same size, this must be provided.
        shuffle: Whether to shuffle the data. Default: True.
                 If set to False, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        interpolation: String, the interpolation method used when resizing images.
                       Defaults to `bilinear`. Supports `bilinear`, `nearest`, `bicubic`,
                       `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    # Returns:
        A `tf.data.Dataset` object.
            - If `label_mode` is None, it yields `float32` tensors of shape
                `(batch_size, image_size[0], image_size[1], num_channels)`,
                encoding images (see below for rules regarding `num_channels`).
            - Otherwise, it yields a tuple `(images, labels)`, where `images`
                has shape `(batch_size, image_size[0], image_size[1], num_channels)`,
                and `labels` follows the format described below.

    Rules regarding labels format:
        - if `label_mode` is `int`, the labels are an `int32` tensor of shape
            `(batch_size,)`.
        - if `label_mode` is `binary`, the labels are a `float32` tensor of
            1s and 0s of shape `(batch_size, 1)`.
        - if `label_mode` is `categorial`, the labels are a `float32` tensor
            of shape `(batch_size, num_classes)`, representing a one-hot
            encoding of the class index.

    Rules regarding number of channels in the yielded images:
        - if `color_mode` is `grayscale`,
            there's 1 channel in the image tensors.
        - if `color_mode` is `rgb`,
            there are 3 channel in the image tensors.
        - if `color_mode` is `rgba`,
            there are 4 channel in the image tensors.
    """
    if seed is None:
        seed = np.random.randint(1e6)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    labels = df['class'].tolist()
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            '`labels` argument should be a list/tuple of integer labels, of '
            'the same size as the number of image files in the target '
            'directory. If you wish to infer the labels from the subdirectory '
            'names in the target directory, pass `labels="inferred"`. '
            'If you wish to get a dataset that only contains images '
            '(no labels), pass `label_mode=None`.')
    if label_mode not in {'int', 'categorical', 'binary', None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            'or None. Received: %s' % (label_mode,))
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
            'Received: %s' % (color_mode,))
    interpolation = image_preprocessing.get_interpolation(interpolation)

    num_classes = len(class_names)
    if label_mode == 'binary' and num_classes != 2:
        raise ValueError(
            'When passing `label_mode="binary", there must exactly 2 classes. '
            'Found the following classes: %s' % (class_names,))

    image_paths = df['file_path'].tolist()
    dataset = paths_and_labels_to_dataset(
        image_paths=image_paths,
        image_size=image_size,
        num_channels=num_channels,
        labels=labels,
        label_mode=label_mode,
        num_classes=num_classes,
        interpolation=interpolation)
    if shuffle:
        # Shuffle locally at each iteration
        buffer_size = batch_size * 8 if buffer_size is None else buffer_size
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    # Include file paths for images as attribute.
    dataset.file_paths = image_paths
    return dataset


def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation):
    """Constructs a dataset of images and labels."""
    # TODO(fchollet): consider making num_parallel_calls settable
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(
        lambda x: path_to_image(x, image_size, num_channels, interpolation),
        num_parallel_calls=AUTOTUNE)
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
        img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
    return img_ds
