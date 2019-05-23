from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch


def cross_entropy_with_one_hot(input, target):
    return torch.nn.CrossEntropyLoss()(input, target.argmax(dim=1))
