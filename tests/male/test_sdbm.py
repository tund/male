from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# class A:
#     def __init__(self):
#         print('A')
#
# class B:
#     def __init__(self):
#         print('B')
#
# func = A
# obj = func()

import os
import numpy as np
import scipy.io as spio
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from male.models.deep_learning.rbm import BernoulliBernoulliRBM

from male.callbacks import Display
from male.models.deep_learning.generative.dbm import DBM
from data_config import data_config
# from sytem_config import system_config
# SYSINFO = system_config()
#
# if SYSINFO['display']==False:
#     import matplotlib
#     # matplotlib.use('Agg')
#     matplotlib.use('Qt4Agg')

# from data_config import data_config
data_info = data_config('MNIST')
data_folder = data_info['data_folder']
temp_folder = data_info['temp_folder']

def load_mnist():

    x_train, y_train = load_svmlight_file('%s/mnist' % data_folder, n_features=784)
    x_test, y_test = load_svmlight_file('%s/mnist.t' % data_folder, n_features=784)

    num_train = 1000
    num_test = 100

    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]


    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    return x_train, y_train, x_test, y_test


def load_pretrain_of_Ruslan(folder=None):
    if folder is None:
        folder = '/data1/hungv/Research/WithHung/source/DBM/source/code_DBM'
    vh_filename = 'fullmnistvh.mat'
    po_filename = 'fullmnistpo.mat'

    vh_mat = spio.loadmat("%s/%s" % (folder, vh_filename), squeeze_me=True)
    hidbiases1 = vh_mat['hidbiases']
    visbiases1 = vh_mat['visbiases']
    vishid1 = vh_mat['vishid']

    po_mat = spio.loadmat("%s/%s" % (folder, po_filename), squeeze_me=True)
    hidbiases2 = po_mat['hidbiases']
    labbiases = po_mat['labbiases']
    labhid = po_mat['labhid']
    visbiases2 = po_mat['visbiases']
    vishid2 = po_mat['vishid']
    # b = [visbiases1, hidbiases1+visbiases2, hidbiases2, labbiases]
    vb = [visbiases1, hidbiases1, visbiases2, hidbiases2 / 2.0, hidbiases2 / 2.0, labbiases]
    w = [vishid1, vishid2, labhid.T]
    x = None
    y = None
    for i in range(10):
        mat = spio.loadmat("%s/digit%d.mat" % (folder, i), squeeze_me=True)
        xi = mat['D']
        yi = np.zeros(shape=[xi.shape[0], 1], dtype=np.int)
        yi[:, 0] = i
        # yi = np.zeros(shape=[xi.shape[0], 10], dtype=np.int)
        # yi[:, i] = 1
        if x is None:
            x = xi
            y = yi
        else:
            x = np.vstack((x, xi))
            y = np.vstack((y, yi))
    x = x / 255.0
    return w, vb, x, y

def test_dbm_mnist():
    # data_type = 0 # my mnist data
    data_type = 1 # Ruslan's mnist data
    if data_type == 0:
        x_train, y_train, x_test, y_test = load_mnist()
        # x = np.vstack([x_train, x_test])
        # y = np.concatenate([y_train, y_test])
        pretrained_params = None
    elif data_type == 1:
        w, vb, x_train, y_train = load_pretrain_of_Ruslan()
        pretrained_params = {'w':w, 'vb': vb}

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(1, 2),
                               freq=1,
                               monitor=[{'metrics': ['recon_err', 'val_recon_err'],
                                         'type': 'line',
                                         'labels': ["training recon error", "validation recon error"],
                                         'title': "Reconstruction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        {'metrics': ['free_energy', 'val_free_energy'],
                                         'type': 'line',
                                         'title': "Free Energies",
                                         'xlabel': "epoch",
                                         'ylabel': "energy",
                                         }
                                        ])

    # filter_display = Display(title="Receptive Fields",
    #                          # dpi='auto',
    #                          dpi=None,
    #                          layout=(1, 1),
    #                          figsize=(8, 8),
    #                          freq=1,
    #                          monitor=[{'metrics': ['filters'],
    #                                    'title': "Receptive Fields",
    #                                    'type': 'img',
    #                                    'num_filters': 100,
    #                                    'disp_dim': (28, 28),
    #                                    'tile_shape': (10, 10),
    #                                    },
    #                                   ])

    gen_display = Display(title="Generated data",
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['generated_data'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_filters': 100,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    recon_display = Display(title="Reconstructed data",
                          # dpi='auto',
                          dpi=None,
                          layout=(1, 1),
                          figsize=(8, 8),
                          freq=1,
                          monitor=[{'metrics': ['reconstruction'],
                                    'title': "Reconstructed data",
                                    'type': 'img',
                                    'data': x_train,
                                    'num_filters': 100,
                                    'disp_dim': (28, 28),
                                    'tile_shape': (10, 10),
                                    },
                                   ])

    model_path = "%s/DBM/numtrain%d" % (temp_folder, x_train.shape[0])
    if os.path.isdir(model_path) == False:
        os.makedirs(model_path)



    # this is the setting for RBM trained with PCD
    model = DBM(
        # layers = [x_train.shape[1], 100, 100, 50],
        layers=[x_train.shape[1], 500, 1000, 10],
        layer_types = ['binary', 'binary', 'binary', 'categorical'],
        pretrained_params=pretrained_params,
		#pretrained_params=None,
        batch_size=100,
        num_pretrain_epochs=50,
        num_epochs=300,
        num_mean_field_steps=11,
        num_gibbs_steps=5,
        weight_cost=2e-4,
        random_state=6789,
        learning_rate=0.001,
        # learning_rate=0.05,
        metrics=['recon_err', 'free_energy', 'recon_loglik'],
        callbacks=[learning_display, gen_display, recon_display],
        model_path=model_path,
        verbose=1)

    # model.x_train = x_train
    # model.y_train = y_train
    model.fit(x_train, y_train)

    # print("Train free energy = %.4f" % model.get_free_energy(x_train).mean())
    # print("Test free energy = %.4f" % model.get_free_energy(x_test).mean())
    #
    # print("Train reconstruction likelihood = %.4f" % model.get_reconstruction_loglik(x_train).mean())
    # print("Test reconstruction likelihood = %.4f" % model.get_reconstruction_loglik(x_test).mean())
    #
    # print("Running KNeighborsClassifier...")
    #
    # x_train1 = model.transform(x_train[:num_train])
    # x_test1 = model.transform(x_test[:num_test])
    #
    # clf = KNeighborsClassifier(n_neighbors=1)
    # clf.fit(x_train1, y_train[:num_train])
    #
    # print("Error = %.4f" % (1 - accuracy_score(y_test[:num_test], clf.predict(x_test1))))
#
#
# def test_bbrbm_mnist_reconstruction_data_classification():
#
#     recon_disp = Display(title='Reconstructed data',
#                          dpi=None,
#                          layout=(1,1),
#                          figsize=(8,8),
#                          freq=1,
#                          monitor=[{'metrics':['reconstruction'],
#                                    'title':'Reconstructed data',
#                                    'type':'img',
#                                    'data': x_train,
#                                    'num_images':50,
#                                    'disp_dim': (28, 28),
#                                    'tile_shape':(10, 10)}])
#     #
#
#     model.fit(x_train)
#
#     print('Running test_bbrbm_mnist_classification_data\n')
#     num_train = 10000
#     num_test model = DBM(
#     #     num_hidden=500,
#     #     num_visible=784,
#     #     batch_size=100,
#     #     num_epochs=100,
#     #     momentum_method='sudden',
#     #     learning_rate=0.001,
#     #     weight_cost=2e-4,
#     #     random_state=6789,
#     #     callbacks=[recon_disp],
#     #     metrics=['recon_err'],
#     #     verbose=1)= 1000
#     x_train_transformed = model.transform(x_train[:num_train, :])
#     x_test_transformed = model.transform(x_test[:num_test, :])
#     clf = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
#     clf.fit(x_train_transformed, y_train[:num_train])
#     y_predict = clf.predict(x_test_transformed)
#     print('Accuracy = %0.4f' % accuracy_score(y_test[:num_test], y_predict))
#
#
#     print('Running test_bbrbm_mnist_reconstruction_data\n')
#     import matplotlib.pyplot as plt
#     plt.style.use('ggplot')
#     from male.utils.disp_utils import tile_raster_images
#
#     x_recon = model.get_reconstruction(x_train)
#     x_disp = np.zeros([100, x_train.shape[1]])
#     for i in range(50):
#         x_disp[2*i] = x_train[i, :]
#         x_disp[2*i+1] = x_recon[i, :]
#
#     img = tile_raster_images(x_disp, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1),
#                              scale_rows_to_unit_interval=False, output_pixel_vals=False)
#     plt.figure()
#     _ = plt.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none')
#     plt.title("Reconstucted samples")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()




if __name__ == '__main__':
    # pytest.main([__file__])
    test_dbm_mnist()


