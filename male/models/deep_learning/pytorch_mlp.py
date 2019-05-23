from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from male import PyTorchModel
from male.utils.generic_utils import make_batches
from male.utils.disp_utils import tile_raster_images
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt


class PyTorchMLP(PyTorchModel):
    def __init__(self, model_name='PyTorchMLP', arch='MLPv1', num_hiddens='10;10',
                 act_func='relu', learning_rate=0.1, **kwargs):
        super(PyTorchMLP, self).__init__(model_name=model_name, **kwargs)
        self.arch = arch
        self.num_hiddens = num_hiddens
        self.act_func = act_func
        self.learning_rate = learning_rate

    def _build_model(self, x):
        num_hiddens = [int(i) for i in self.num_hiddens.split(';')]
        act_func = F.relu if self.act_func == 'relu' else F.tanh
        if self.arch == 'MLPv1':
            self.net = MLPv1(data_dim=x.shape[1], num_hiddens=num_hiddens,
                             num_classes=self.num_classes, act_func=act_func)
        else:
            self.net = MLPv2(data_dim=x.shape[1], num_hiddens=num_hiddens,
                             num_classes=self.num_classes, act_func=act_func)
        self.criterion = nn.CrossEntropyLoss()
        self.optz = optim.SGD(self.net.parameters(), lr=self.learning_rate)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        batches = make_batches(x.shape[0], self.batch_size)
        # convert data to Torch Tensor
        x_tensor = torch.from_numpy(x).type(self.float).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)
        if x_valid is not None:
            x_valid_tensor = torch.from_numpy(x_valid).type(self.float).to(self.device)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x_tensor[batch_start:batch_end]
                y_batch = y_tensor[batch_start:batch_end]
                y_batch_np = y[batch_start:batch_end]

                self.optz.zero_grad()   # zero the gradient buffers

                y_pred = self.net(x_batch)
                loss = self.criterion(y_pred, y_batch)
                batch_logs.update({'loss': loss.item()})

                # compute the gradients
                loss.backward()

                self.optz.step()  # does the update

                batch_logs.update(self._on_batch_end(x_batch, y_batch_np, logs=batch_logs))

                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid_tensor, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def predict(self, x):
        check_is_fitted(self, 'net')
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).type(self.float).to(self.device)
            y = self.forward(x).argmax(dim=1)
            return y.cpu().detach().numpy()

    def predict_proba(self, x):
        check_is_fitted(self, 'net')
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).type(self.float).to(self.device)
            return F.softmax(self.forward(x)).cpu().detach().numpy()

    def display(self, param, **kwargs):
        if param == 'weights':
            self.disp_weights(**kwargs)
        else:
            raise NotImplementedError

    def disp_weights(self, disp_dim=None, tile_shape=None,
                     output_pixel_vals=False, **kwargs):
        w = self.net.layers[0].weight.data.cpu().detach().numpy().T
        if w.ndim < 2:
            w = w[..., np.newaxis]

        if disp_dim is None:
            n = int(np.sqrt(w.shape[0]))
            disp_dim = (n, n)
        else:
            assert len(disp_dim) == 2
        n = np.prod(disp_dim)

        if tile_shape is None:
            tile_shape = (w.shape[1], 1)
        assert w.shape[1] == np.prod(tile_shape)

        img = tile_raster_images(w.T, img_shape=disp_dim, tile_shape=tile_shape,
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=output_pixel_vals)

        if 'ax' in kwargs:
            ax = kwargs['ax']
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            ax.grid(0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("epoch #{}".format(kwargs['epoch']), fontsize=28)
        else:
            fig, ax = plt.subplots()
            ax.set_title(kwargs['title'] if 'title' in kwargs else "Learned weights",
                         fontsize=28)
            ax.axis('off')
            plt.colorbar()
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            plt.show()


class MLPv1(nn.Module):
    def __init__(self, data_dim, num_hiddens, num_classes, act_func):
        super(MLPv1, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(data_dim, num_hiddens[0])])
        for i in range(1, len(num_hiddens)):
            self.layers.append(nn.Linear(num_hiddens[i - 1], num_hiddens[i]))
        self.layers.append(nn.Linear(num_hiddens[-1], num_classes))
        self.act_func = act_func

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = self.act_func(x)
        return x


class MLPv2(nn.Module):
    def __init__(self, data_dim, num_hiddens, num_classes, act_func):
        super(MLPv2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(data_dim, num_hiddens[0]),
            nn.ReLU(),
        )
        for i in range(1, len(num_hiddens)):
            self.layers.add_module('fc{}'.format(i), nn.Linear(num_hiddens[i - 1], num_hiddens[i]))
            self.layers.add_module('relu{}'.format(i), nn.ReLU())
        self.layers.add_module('fc_out', nn.Linear(num_hiddens[-1], num_classes))        

    def forward(self, x):        
        return self.layers(x)
