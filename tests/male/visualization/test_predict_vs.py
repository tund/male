from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from sklearn.datasets import load_svmlight_file

import numpy as np

from male.models.kernel import KSGD
from male.utils.disp_utils import visualize_classification_prediction

data_dir = 'C:/Data/2d/'
data_name = 'train.scale.txt'
n_features = 2
# NOTE: SGD have some bugs with binary datasets

file_name = data_dir + data_name

x_train, y_train = load_svmlight_file(file_name, n_features=2)
x_train = x_train.toarray()
N = x_train.shape[0]
print('N =', N)
idx_train = np.random.permutation(N)
x_train = x_train[idx_train]
y_train = y_train[idx_train]

learner = KSGD(lbd=0.001,
               eps=0.001,
               gamma=20,
               kernel='gaussian',
               loss='hinge',
               batch_size=1,
               avg_weight=False)

visualize_classification_prediction(learner, x_train, y_train, grid_size=500)
