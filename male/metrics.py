from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import six
import math
import warnings
import numpy as np
from scipy import linalg
from sklearn import metrics

from .configs import model_dir
from .utils.data_utils import get_file
from .utils.generic_utils import make_batches


def auc(y_true, y_pred, pos_label=1):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=pos_label)
    return metrics.auc(fpr, tpr)


class InceptionMetric(object):

    def __init__(self, inception_graph=None, name="", **kwargs):
        self.name = name
        self.inception_graph = inception_graph

    def _init_inception(self):
        import tarfile
        import tensorflow as tf
        from six.moves import urllib

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
        self.inception_graph = tf.Graph()
        with self.inception_graph.as_default():
            with tf.gfile.FastGFile(os.path.join(
                    MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

    def _set_graph(self, graph):
        self.inception_graph = graph

    def on_score_begin(self, num_data=50000):
        pass

    def on_batch_end(self, batch_start, batch_end, batch_out, **kwargs):
        pass

    def on_score_end(self, **kwargs):
        pass

    def score(self, images, batch_size=32, **kwargs):
        pass

    def _get_params_to_dump(self, deep=True):
        out = dict()
        for key, value in six.iteritems(self.__dict__):
            if ((not type(value).__module__.startswith('tf')) and
                    (not type(value).__module__.startswith('tensorflow')) and
                    (key != 'best_params')):
                out[key] = value
        # param_names = ['tf_graph', 'tf_config', 'tf_merged_summaries']
        # for key in param_names:
        #     if key in self.__dict__:
        #         out[key] = self.__getattribute__(key)
        return out

    def __getstate__(self):
        from . import __version__
        out = self._get_params_to_dump(deep=True)
        if type(self).__module__.startswith('male.'):
            return dict(out, _male_version=__version__)
        else:
            return out


class InceptionMetricList(InceptionMetric):

    def __init__(self, metrics=[], **kwargs):
        super(InceptionMetricList, self).__init__(**kwargs)
        self.metrics = [m for m in metrics]

    def append(self, m):
        self.metrics.append(m)

    def _set_graph(self, graph):
        for m in self.metrics:
            m._set_graph(graph)

    def on_score_begin(self, num_data=50000):
        for m in self.metrics:
            m.on_score_begin(num_data=num_data)

    def on_batch_end(self, batch_start, batch_end, batch_out, **kwargs):
        for (i, m) in enumerate(self.metrics):
            m.on_batch_end(batch_start, batch_end, batch_out[i], **kwargs)

    def on_score_end(self, **kwargs):
        score_out = []
        for m in self.metrics:
            score_out.append(m.on_score_end(**kwargs))
        return score_out

    def score(self, images, batch_size=32, **kwargs):
        imgs = images.copy().astype(np.float32)
        if np.max(imgs) <= 1.0:
            imgs *= 255.0

        if self.inception_graph is None:
            self._init_inception()
            self._set_graph(self.inception_graph)
            for m in self.metrics:
                m._init_inception_layer()

        import tensorflow as tf
        from .backend import tensorflow_backend as tf_backend

        self.on_score_begin(num_data=imgs.shape[0])
        # Works with an arbitrary minibatch size.
        with tf.Session(graph=self.inception_graph,
                        config=tf_backend.get_default_config()) as sess:
            batches = make_batches(imgs.shape[0], batch_size)
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                pred = sess.run([m.inception_layer for m in self.metrics],
                                {'ExpandDims:0': imgs[batch_start:batch_end]})
                self.on_batch_end(batch_start, batch_end, pred)
        return self.on_score_end()

    def get_score_dict(self, scores):
        score = {}
        for (m, s) in zip(self.metrics, scores):
            if isinstance(m, InceptionScore):
                score[m.name] = s[0]
                score["{}_std".format(m.name)] = s[1]
            else:
                score[m.name] = s
        return score


class InceptionScore(InceptionMetric):

    def __init__(self, inception_layer=None, name="inception_score", **kwargs):
        super(InceptionScore, self).__init__(name=name, **kwargs)
        self.inception_layer = inception_layer
        self.preds = None

    def _init_inception_layer(self):
        import tensorflow as tf
        from .backend import tensorflow_backend as tf_backend
        with tf.Session(graph=self.inception_graph,
                        config=tf_backend.get_default_config()) as sess:
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            if pool3.get_shape().as_list()[0] is not None:
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
                        o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits = tf.matmul(tf.squeeze(pool3, axis=[1, 2]), w)
            self.inception_layer = tf.nn.softmax(logits)

    # Call this function with list of images. Each of elements should be a
    # numpy array with values ranging from 0 to 255.
    def score(self, images, batch_size=32, **kwargs):
        imgs = images.copy().astype(np.float32)
        if np.max(imgs) <= 1.0:
            imgs *= 255.0

        if self.inception_layer is None:
            self._init_inception()
            self._init_inception_layer()

        import tensorflow as tf
        from .backend import tensorflow_backend as tf_backend

        self.on_score_begin(num_data=imgs.shape[0])
        # Works with an arbitrary minibatch size.
        with tf.Session(graph=self.inception_graph,
                        config=tf_backend.get_default_config()) as sess:

            batches = make_batches(imgs.shape[0], batch_size)
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                pred = sess.run(self.inception_layer, {'ExpandDims:0': imgs[batch_start:batch_end]})
                self.on_batch_end(batch_start, batch_end, pred)

        return self.on_score_end(**kwargs)

    def on_score_begin(self, num_data=50000):
        self.preds = np.zeros([num_data, 1008])

    def on_batch_end(self, batch_start, batch_end, batch_out, **kwargs):
        self.preds[batch_start:batch_end] = batch_out

    def on_score_end(self, splits=10):
        scores = []
        for i in range(splits):
            part = self.preds[(i * self.preds.shape[0] // splits)
                              :((i + 1) * self.preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        self.preds = None
        return np.mean(scores), np.std(scores)


class FID(InceptionMetric):
    """Frechet Inception Distance
    """

    precalc_stats_info = {
        "celeba": {"filename": "fid_stats_celeba.npz",
                   "url": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz"},
        "lsun": {"filename": "fid_stats_lsun_train.npz",
                 "url": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz"},
        "cifar10": {"filename": "fid_stats_cifar10_train.npz",
                    "url": "http://bioinf.jku.at/research/ttur/ttur_stats"
                           "/fid_stats_cifar10_train.npz"},
        "shvn": {"filename": "fid_stats_svhn_train.npz",
                 "url": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_svhn_train.npz"},
        "imagenet": {"filename": "fid_stats_imagenet_train.npz",
                     "url": "http://bioinf.jku.at/research/ttur/ttur_stats"
                            "/fid_stats_imagenet_train.npz"}
    }

    def __init__(self, data=None, inception_layer=None, name="FID", **kwargs):
        super(FID, self).__init__(name=name, **kwargs)
        self.data = data
        self.mean = None
        self.std = None
        self.inception_layer = inception_layer
        self.preds = None

    def _init_inception_layer(self):
        import tensorflow as tf
        from .backend import tensorflow_backend as tf_backend

        with tf.Session(graph=self.inception_graph,
                        config=tf_backend.get_default_config()) as sess:
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            if pool3.get_shape().as_list()[0] is not None:
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
                        o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
            self.inception_layer = pool3

        self._set_stats()

    def _set_stats(self):
        if isinstance(self.data, str):
            fpath = get_file(FID.precalc_stats_info[self.data.lower()]["filename"],
                             origin=FID.precalc_stats_info[self.data.lower()]["url"],
                             cache_subdir="FID")
            # load precalculated training set statistics
            f = np.load(fpath)
            self.mean, self.std = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            self.mean, self.std = self.calculate_activation_statistics(self.data)

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                 inception net ( like returned by the function 'get_predictions')
                 for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                   on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                   generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                   precalcualted on an representive data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def calculate_activation_statistics(self, images, batch_size=32):
        """Calculation of the statistics used by the FID.
        Params:
            -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                             must lie between 0 and 255.
            -- batch_size  : the images numpy array is split into batches with batch size
                             batch_size. A reasonable batch size depends on the available hardware.
        Returns:
            -- mu    : The mean over samples of the activations of the pool_3 layer of
                       the inception model.
            -- sigma : The covariance matrix of the activations of the pool_3 layer of
                       the inception model.
        """
        imgs = images.copy().astype(np.float32)
        if np.max(imgs) <= 1.0:
            imgs *= 255.0

        if self.inception_layer is None:
            self._init_inception()
            self._init_inception_layer()

        self.activations(imgs, batch_size=batch_size)
        mu = np.mean(self.preds, axis=0)
        sigma = np.cov(self.preds, rowvar=False)
        self.preds = None
        return mu, sigma

    def activations(self, images, batch_size=32):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
            -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                             must lie between 0 and 256.
            -- batch_size  : the images numpy array is split into batches with batch size
                             batch_size. A reasonable batch size depends on the disposable hardware.
        Returns:
            -- A numpy array of dimension (num images, 2048) that contains the
               activations of the given tensor when feeding inception with the query tensor.
        """

        import tensorflow as tf
        from .backend import tensorflow_backend as tf_backend
        self.on_score_begin(num_data=images.shape[0])
        batches = make_batches(images.shape[0], batch_size)
        with tf.Session(graph=self.inception_graph,
                        config=tf_backend.get_default_config()) as sess:
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                pred = sess.run(self.inception_layer,
                                {'ExpandDims:0': images[batch_start:batch_end]})
                self.on_batch_end(batch_start, batch_end, pred)

    def on_score_begin(self, num_data=50000):
        self.preds = np.zeros([num_data, 2048])

    def on_batch_end(self, batch_start, batch_end, batch_out, **kwargs):
        self.preds[batch_start:batch_end] = batch_out.reshape(batch_end - batch_start, -1)

    def on_score_end(self):
        mu = np.mean(self.preds, axis=0)
        sigma = np.cov(self.preds, rowvar=False)
        self.preds = None
        return FID.calculate_frechet_distance(self.mean, self.std, mu, sigma)

    def score(self, images, batch_size=32, **kwargs):
        imgs = images.copy().astype(np.float32)
        if np.max(imgs) <= 1.0:
            imgs *= 255.0

        if self.inception_layer is None:
            self._init_inception()
            self._init_inception_layer()

        self.activations(imgs, batch_size=batch_size)
        return self.on_score_end()
