from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import math
import numpy as np
from sklearn import metrics

from .configs import model_dir


def auc(y_true, y_pred, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


class InceptionScore(object):
    inception_model = None
    inception_graph = None

    @staticmethod
    def init_inception():
        import tarfile
        import tensorflow as tf
        from six.moves import urllib
        from .backend import tensorflow_backend as tf_backend

        MODEL_DIR = os.path.join(model_dir(), "ImageNet")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
        InceptionScore.inception_graph = tf.Graph()
        with InceptionScore.inception_graph.as_default():
            with tf.gfile.FastGFile(os.path.join(
                    MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

        tf_config = tf_backend.get_default_config()

        # Works with an arbitrary minibatch size.
        with tf.Session(graph=InceptionScore.inception_graph, config=tf_config) as sess:
            # pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            ops = pool3.graph.get_operations()
            for op_idx, op in enumerate(ops):
                for o in op.outputs:
                    shape = o.get_shape()
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o._shape = tf.TensorShape(new_shape)
            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits = tf.matmul(tf.squeeze(pool3), w)
            InceptionScore.inception_model = tf.nn.softmax(logits)

            # return softmax, inception_graph

    # Call this function with list of images. Each of elements should be a
    # numpy array with values ranging from 0 to 255.
    @staticmethod
    def inception_score(images, splits=10):
        if InceptionScore.inception_model is None:
            InceptionScore.init_inception()

        import tensorflow as tf
        from .backend import tensorflow_backend as tf_backend

        assert (type(images) == list)
        assert (type(images[0]) == np.ndarray)
        assert (len(images[0].shape) == 3)
        assert (np.max(images[0]) > 10)
        assert (np.min(images[0]) >= 0.0)
        inps = []
        for img in images:
            img = img.astype(np.float32)
            inps.append(np.expand_dims(img, 0))
        bs = 100

        tf_config = tf_backend.get_default_config()

        with tf.Session(graph=InceptionScore.inception_graph, config=tf_config) as sess:
            preds = []
            n_batches = int(math.ceil(float(len(inps)) / float(bs)))
            for i in range(n_batches):
                # sys.stdout.write(".")
                # sys.stdout.flush()
                inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
                inp = np.concatenate(inp, 0)
                pred = sess.run(InceptionScore.inception_model, {'ExpandDims:0': inp})
                preds.append(pred)
            preds = np.concatenate(preds, 0)
            scores = []
            for i in range(splits):
                part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                kl = np.mean(np.sum(kl, 1))
                scores.append(np.exp(kl))
            return np.mean(scores), np.std(scores)
