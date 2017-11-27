from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import TensorFlowModel


class Word2Vec(TensorFlowModel):
    """Word Embedding
    """

    def __init__(self,
                 model_name="Word2Vec",
                 emb_size=128,
                 vocab_size=10000,
                 window=5,
                 num_skips=2,
                 num_neg_samples=5,
                 learning_rate=0.025,
                 **kwargs):
        super(Word2Vec, self).__init__(model_name=model_name, **kwargs)
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.window = window
        self.num_skips = num_skips
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate

    def _init(self):
        super(Word2Vec, self)._init()

    def _init_params(self, x):
        super(Word2Vec, self)._init_params(x)
        self.num_words = 0
        self.data_index = 0
        self.dictionary = None
        self.reverse_dictionary = None
        self.final_embeddings = None

    def _build_model(self, x):
        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.y = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        # Look up embeddings for inputs.
        with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.emb_size],
                                                       -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self.x)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm

        # Construct the variables for the NCE loss
        with tf.name_scope("nce"):
            nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.emb_size],
                                                          stddev=1.0 / np.sqrt(self.emb_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                      biases=nce_biases,
                                                      labels=self.y,
                                                      inputs=embed,
                                                      num_sampled=self.num_neg_samples,
                                                      num_classes=self.vocab_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope("optimizer"):
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            grads = optimizer.compute_gradients(self.loss, var_list=params)
            self.train_op = optimizer.apply_gradients(grads)

            for var in params:
                tf.summary.histogram(var.op.name + '/values', var)

            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

                    # Compute the cosine similarity between minibatch examples and all embeddings.
                    # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                    # normalized_embeddings = embeddings / norm
                    # valid_embeddings = tf.nn.embedding_lookup(
                    #     normalized_embeddings, valid_dataset)
                    # similarity = tf.matmul(
                    #     valid_embeddings, normalized_embeddings, transpose_b=True)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        # Filling 4 global variables:
        # data - list of codes (integers from 0 to vocabulary_size-1).
        #   This is the original text but words are replaced by their codes
        # count - map of words(strings) to count of occurrences
        # dictionary - map of words(strings) to their codes(integers)
        # reverse_dictionary - maps codes(integers) to words(strings)
        data, count, self.dictionary, self.reverse_dictionary = Word2Vec.build_dataset(
            tf.compat.as_str(x).split(), self.vocab_size)
        self.vocab_size = len(self.dictionary)
        self.num_words = len(data)
        callbacks._update_params({'num_samples': self.num_words})

        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            num_batches = int(math.ceil((self.num_words - self.data_index - 2 * self.window)
                                        / (self.batch_size // self.num_skips)))
            for batch_idx in range(num_batches):
                batch_logs = {'batch': batch_idx,
                              'size': self.batch_size}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                self.data_index, x_batch, y_batch = Word2Vec.generate_batch(
                    data, self.data_index, self.batch_size, self.num_skips, self.window)

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss = self.tf_session.run([self.train_op, self.loss],
                                              feed_dict={self.x: x_batch, self.y: y_batch})

                batch_logs['loss'] = loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _on_train_end(self):
        super(Word2Vec, self)._on_train_end()
        self.final_embeddings = self.tf_session.run(self.normalized_embeddings)

    @staticmethod
    def build_dataset(words, n_words):
        """Process raw inputs into a dataset."""
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

    # Step 3: Function to generate a training batch for the skip-gram model.
    @staticmethod
    def generate_batch(data, data_index, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if data_index == len(data):
                buffer = data[:span]
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return data_index, batch, labels

    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.
    def plot_with_labels(self, low_dim_embs, labels, **kwargs):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        if 'ax' in kwargs:
            ax = kwargs['ax']
            for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                ax.scatter(x, y)
                ax.annotate(label,
                            xy=(x, y),
                            xytext=(5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom')

    def disp_embedding(self, num_words=500, **kwargs):
        sess = self._get_session()
        embs = sess.run(self.normalized_embeddings)
        if sess != self.tf_session:
            sess.close()

        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        low_dim_embs = tsne.fit_transform(embs[:num_words, :])
        labels = [self.reverse_dictionary[i] for i in range(num_words)]
        self.plot_with_labels(low_dim_embs, labels, **kwargs)

    def display(self, param, **kwargs):
        if param == 'embedding':
            self.disp_embedding(**kwargs)
        else:
            raise NotImplementedError
