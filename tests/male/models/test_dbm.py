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

def test_dbm_mnist():
    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

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
        layers=[x_train.shape[1], 500],
        batch_size=100,
        num_pretrain_epochs=50,
        num_epochs=300,
        num_mean_field_steps=30,
        num_gibbs_steps=15,
        weight_cost=2e-4,
        random_state=6789,
        learning_rate=0.001,
        # learning_rate=0.05,
        metrics=['recon_err', 'free_energy', 'recon_loglik'],
        callbacks=[learning_display, gen_display, recon_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        model_path=model_path,
        verbose=1)


    model.fit(x)

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


