from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np

from male import PyTorchModel
from male.models.linear import GLM
from male.utils.generic_utils import make_batches
from male.utils.disp_utils import tile_raster_images

import matplotlib.pyplot as plt

plt.style.use('ggplot')

EPS = np.finfo(np.float32).eps


class PyTorchGLM(PyTorchModel, GLM):
    '''Generalized Linear Model using PyTorch
    '''

    def __init__(self,
                 model_name='PyTorchGLM',
                 learning_rate=0.01,
                 **kwargs):
        super(PyTorchGLM, self).__init__(model_name=model_name, **kwargs)
        self.learning_rate = learning_rate

    def _init_params(self, x):
        # initialize weights
        if self.num_classes > 2:
            self.w = torch.randn(x.shape[1], self.num_classes,
                                 device=self.device, dtype=self.float, requires_grad=True)
            with torch.no_grad():
                self.w.mul_(0.01)  # multiply with standard deviation
            self.b = torch.zeros(self.num_classes,
                                 device=self.device, dtype=self.float, requires_grad=True)
        else:
            self.w = torch.randn(x.shape[1], 1,
                                 device=self.device, dtype=self.float, requires_grad=True)
            with torch.no_grad():
                self.w.mul_(0.01)  # multiply with standard deviation
            self.b = torch.zeros(1,
                                 device=self.device, dtype=self.float, requires_grad=True)

    def _cross_entropy_with_one_hot(self, input, target):
        return torch.nn.CrossEntropyLoss()(input, target.argmax(dim=1))

    def _build_model(self, x):
        if self.loss == 'logit':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.loss == 'softmax':
            self.criterion = self._cross_entropy_with_one_hot
        else:  # quadratic (regression)
            self.criterion = torch.nn.MSELoss()
        self.regularization = 0.0
        if self.l2_penalty > 0:
            self.regularization += 0.5 * self.l2_penalty * torch.sum(torch.pow(self.w, 2))
        if self.l1_penalty > 0:
            self.regularization += self.l1_penalty * torch.reduce_sum(torch.abs(self.w))

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        batches = make_batches(x.shape[0], self.batch_size)
        # convert data to Torch Tensor
        x_tensor = torch.from_numpy(x).to(self.device)
        y_tensor = torch.from_numpy(y).to(self.device)
        if x_valid is not None:
            x_valid_tensor = torch.from_numpy(x_valid).to(self.device)
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

                y_pred = x_batch.mm(self.w) + self.b
                y_pred = y_pred.squeeze()
                loss = self.criterion(y_pred, y_batch)
                total_loss = loss + self.regularization

                # compute the gradients
                total_loss.backward()

                # update parameters & reset gradients
                with torch.no_grad():
                    self.w -= self.learning_rate * self.w.grad
                    self.b -= self.learning_rate * self.b.grad
                    self.w.grad.zero_()
                    self.b.grad.zero_()

                batch_logs.update(self._on_batch_end(x_batch, y_batch_np))

                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid_tensor, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def get_loss(self, x, y, **kwargs):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).to(self.device)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y).to(self.device)
            y_pred = x.mm(self.w) + self.b
            return self.criterion(y_pred, y).item()

    def get_link(self, x, **kwargs):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).to(self.device)
            y = x.mm(self.w) + self.b
            if self.link == 'logit':
                y = torch.sigmoid(y)
            elif self.link == 'softmax':
                y = torch.nn.functional.softmax(y)
            return y.cpu().detach().numpy()

    def disp_weights(self, disp_dim=None, tile_shape=None,
                     output_pixel_vals=False, **kwargs):
        w = self.w.cpu().detach().numpy()
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
