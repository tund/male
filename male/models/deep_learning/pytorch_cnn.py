from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms

from male import PyTorchModel
from male.configs import data_dir
from male.utils.generic_utils import make_batches
from male.utils.disp_utils import tile_raster_images
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt


class PyTorchConvNet(PyTorchModel):
    def __init__(self, model_name='PyTorchConvNet', dataset='cifar10',
                 learning_rate=0.1, momentum=0.9, **kwargs):
        super(PyTorchConvNet, self).__init__(model_name=model_name, **kwargs)
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.momentum = momentum

    def _build_model(self, x):
        if self.dataset == 'cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(),  # convert to PIL-Image of range [0, 1]
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # normalize to [-1, 1]
            )
            self.trainset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir(), 'cifar10'),
                                                         train=True, download=True, transform=transform)
            self.trainset.data = self.trainset.data[:1000]
            self.trainset.targets = self.trainset.targets[:1000]
            if self.verbose > 0:
                print(self.trainset)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                           shuffle=True, num_workers=2)

            testset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir(), 'cifar10'),
                                                   train=False, download=True, transform=transform)
            testset.data = testset.data[:100]
            testset.targets = testset.targets[:100]
            if self.verbose > 0:
                print(testset)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                          shuffle=False, num_workers=2)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        else:
            raise NotImplementedError

        self.net = ConvNet().to(self.device)
        if self.verbose > 0:
            print(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.optz = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        callbacks._update_params({'num_samples': len(self.trainset)})
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, data in enumerate(self.trainloader, 0):
                x_batch, y_batch = data  # get the input data
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                batch_logs = {'batch': batch_idx,
                              'size': y_batch.size(0)}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                self.optz.zero_grad()   # zero the gradient buffers

                y_pred = self.net(x_batch)  # forward
                loss = self.criterion(y_pred, y_batch)  # loss
                batch_logs.update({'loss': loss.item()})  # update batch history

                loss.backward()  # backward: compute the gradients

                self.optz.step()  # does the update

                batch_logs.update(self._on_batch_end(x_batch, y_batch, logs=batch_logs))

                callbacks.on_batch_end(batch_idx, batch_logs)

            '''
            if do_validation:
                outs = self._on_batch_end(x_valid_tensor, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value
            '''

            self._on_epoch_end()
            callbacks.on_epoch_end(self.epoch - 1, epoch_logs)

    def _on_batch_end(self, x, y=None, logs={}):
        outs = {}
        for m in self.metrics:
            if m in logs:
                continue
            if m == 'loss':
                outs.update({m: self.get_loss(x, y)})
            if m == 'acc':
                outs.update({m: self.score(x, y)})
            if m == 'err':
                if self.task == 'classification':
                    outs.update({m: 1 - self.score(x, y)})
                else:
                    outs.update({m: -self.score(x, y)})
        return outs

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

    def acc_test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
