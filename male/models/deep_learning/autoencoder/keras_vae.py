from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
import numpy as np

import tensorflow as tf
from keras import objectives
from keras import backend as K
from keras.models import Model as KModel
from keras.models import model_from_json
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True
set_session(tf.Session(config=config))

from ....model import Model
from ....utils.disp_utils import tile_raster_images
from ....utils.io_utils import ask_to_proceed_with_overwrite

import matplotlib.pyplot as plt


class KerasVAE(Model):
    """Variational Autoencoder in Keras
    """

    def __init__(self,
                 model_name="KerasVAE",
                 num_visible=500,
                 optimizer='adagrad',
                 num_z=50, z_init=0.01,
                 num_hiddens=[], act_funcs=[],
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(KerasVAE, self).__init__(**kwargs)
        self.num_visible = num_visible
        self.num_hiddens = num_hiddens
        self.act_funcs = act_funcs
        self.num_z = num_z
        self.z_init = z_init
        self.optimizer = optimizer

    def _init(self):
        super(KerasVAE, self)._init()

        self.x_ = Input(batch_shape=[None, self.num_visible])
        self.h_ = []
        for (i, num_h) in enumerate(self.num_hiddens):
            if i == 0:
                self.h_ += [Dense(num_h, activation=self.act_funcs[i])(self.x_)]
            else:
                self.h_ += [Dense(num_h, activation=self.act_funcs[i])(self.h_[-1])]
        if self.h_:
            self.z_mean_ = Dense(self.num_z)(self.h_[-1])
            self.z_log_std_ = Dense(self.num_z)(self.h_[-1])
        else:
            self.z_mean_ = Dense(self.num_z)(self.x_)
            self.z_log_std_ = Dense(self.num_z)(self.x_)

        self.z_ = Lambda(self.sampling, output_shape=(self.num_z,))([self.z_mean_, self.z_log_std_])

        self.gen_h_ = []
        for (i, num_h) in reversed(list(enumerate(self.num_hiddens))):
            if i == len(self.num_hiddens) - 1:
                self.gen_h_ += [Dense(num_h, activation=self.act_funcs[i])(self.z_)]
            else:
                self.gen_h_ += [Dense(num_h, activation=self.act_funcs[i])(self.gen_h_[-1])]
        if self.gen_h_:
            self.gen_x_ = Dense(self.num_visible, activation='sigmoid')(self.gen_h_[-1])
        else:
            self.gen_x_ = Dense(self.num_visible, activation='sigmoid')(self.z_)

        # the network consisting of encoder and generator
        self.model_ = KModel(self.x_, self.gen_x_)

        # encoder
        self.encoder_ = KModel(self.x_, self.z_mean_)

        # decoder
        self.decode_z_ = Input(shape=[self.num_z, ])
        self.decode_h_ = []
        for i in range(len(self.num_hiddens)):
            if i == 0:
                self.decode_h_ += [
                    self.model_.layers[len(self.num_hiddens) + i + 4](self.decode_z_)]
            else:
                self.decode_h_ += [
                    self.model_.layers[len(self.num_hiddens) + i + 4](self.decode_h_[-1])]
        if self.decode_h_:
            self.decode_x_ = self.model_.layers[2 * len(self.num_hiddens) + 4](self.decode_h_[-1])
        else:
            self.decode_x_ = self.model_.layers[len(self.num_hiddens) + 4](self.decode_z_)
        self.decoder_ = KModel(self.decode_z_, self.decode_x_)

        self.model_.compile(optimizer=self.optimizer, loss=self.loss)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        x = x.astype(np.float32)

        if do_validation:
            x_valid = x_valid.astype(np.float32)
            self.model_.fit(x, x, shuffle=True,
                            nb_epoch=self.num_epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_valid, x_valid))
        else:
            self.model_.fit(x, x, shuffle=True,
                            nb_epoch=self.num_epochs,
                            batch_size=self.batch_size)
        self.epoch_ = self.num_epochs

    def sampling(self, args):
        z_mean, z_log_std = args
        e = K.random_normal(shape=(self.num_z,), mean=0.0, std=self.z_init)
        return z_mean + K.exp(z_log_std / 2) * e

    def loss(self, x, r):
        xent_loss = K.sum(K.binary_crossentropy(r, x), axis=1)
        kl_loss = - 0.5 * K.sum(
            1 + self.z_log_std_ - K.square(self.z_mean_) - K.exp(self.z_log_std_), axis=1)
        return K.mean(xent_loss + kl_loss)

    def get_encoding(self, x):
        return self.encoder_.predict(x, batch_size=self.batch_size)

    def get_reconstruction(self, x):
        return self.decoder_.predict(self.get_encoding(x), batch_size=self.batch_size)

    def get_loss(self, x, y, *args, **kwargs):
        return self.model_.evaluate(x, x, batch_size=self.batch_size)

    def get_xent_loss(self, x):
        r = self.get_reconstruction(x)
        return K.eval(K.binary_crossentropy(r, x))

    def get_reconstruction_error(self, x):
        r = self.get_reconstruction(x)
        return np.abs(x - r)

    def disp_params(self, params, num_filters=100, filter_idx=None,
                    disp_dim=None, tile_shape=(10, 10), output_pixel_vals=False):
        if params == 'weights':
            if disp_dim is None:
                n = int(np.sqrt(self.num_visible))
                disp_dim = (n, n)
            else:
                assert len(disp_dim) == 2
            n = np.prod(disp_dim)

            assert num_filters == np.prod(tile_shape)

            if filter_idx is None:
                filter_idx = np.random.permutation(
                    self.num_hiddens[0] if len(self.num_hiddens) > 0 else self.num_z
                )[:num_filters]
            # w = self.model_.layers[1].get_weights()[0].T[filter_idx, :N]
            w = self.model_.layers[-1].get_weights()[0][filter_idx, :n]
            img = tile_raster_images(w, img_shape=disp_dim, tile_shape=tile_shape,
                                     tile_spacing=(1, 1),
                                     scale_rows_to_unit_interval=False,
                                     output_pixel_vals=output_pixel_vals)
            fig, ax = plt.subplots()
            _ = ax.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none')
            ax.set_title("Filters of the first layer @ epoch #{}".format(self.epoch_))
            plt.colorbar()
            ax.axis('off')
            plt.show()
        else:
            raise NotImplementedError

    def _save_model(self, file_path, overwrite=True):
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(file_path):
            proceed = ask_to_proceed_with_overwrite(file_path)
            if not proceed:
                return
        open("{}/{}_model.json".format(file_path, self.epoch_), 'w').write(
            self.model_.to_json())
        open("{}/{}_encoder.json".format(file_path, self.epoch_), 'w').write(
            self.encoder_.to_json())
        open("{}/{}_decoder.json".format(file_path, self.epoch_), 'w').write(
            self.decoder_.to_json())
        self.model_.save_weights("{}/{}_model.h5".format(file_path, self.epoch_),
                                 overwrite=True)
        self.encoder_.save_weights("{}/{}_encoder.h5".format(file_path, self.epoch_),
                                   overwrite=True)
        self.decoder_.save_weights("{}/{}_decoder.h5".format(file_path, self.epoch_),
                                   overwrite=True)

    def _load_model(self, file_path):
        # to load json file: model = model_from_json(open(file_path).read())
        self.model_.load_weights(file_path + '_model_h5')
        self.encoder_.load_weights(file_path + '_encoder_h5')
        self.decoder_.load_weights(file_path + '_decoder_h5')
        return self

    def get_params(self, deep=True):
        out = super(KerasVAE, self).get_params(deep=deep)
        out.update({'num_z': self.num_z,
                    'z_init': self.z_init,
                    'optimizer': self.optimizer,
                    'num_visible': self.num_visible,
                    'num_hiddens': copy.deepcopy(self.num_hiddens),
                    'act_funcs': copy.deepcopy(self.act_funcs),
                    })
        return out

    def get_all_params(self, deep=True):
        out = self.get_params(deep=deep)
        out.update({'x_': copy.deepcopy(self.x_),
                    'h_': copy.deepcopy(self.h_),
                    'z_mean_': copy.deepcopy(self.z_mean_),
                    'z_log_std_': copy.deepcopy(self.z_log_std_),
                    'z_': copy.deepcopy(self.z_),
                    'gen_h_': copy.deepcopy(self.gen_h_),
                    'gen_x_': copy.deepcopy(self.gen_x_),
                    'model_': copy.deepcopy(self.model_),
                    'encoder_': copy.deepcopy(self.encoder_),
                    'decode_z_': copy.deepcopy(self.decode_z_),
                    'decode_h_': copy.deepcopy(self.decode_h_),
                    'decode_x_': copy.deepcopy(self.decode_x_),
                    'decoder_': copy.deepcopy(self.decoder_),
                    })
        return out
