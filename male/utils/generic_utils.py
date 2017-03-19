from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import six
import time
import copy
import warnings
import numpy as np


def tuid():
    '''
    Create a string ID based on current time
    :return: a string formatted using current time
    '''
    return time.strftime('%Y-%m-%d_%H.%M.%S')


def deepcopy(obj):
    try:
        return copy.deepcopy(obj)
    except:
        warnings.warn("Fail to deepcopy {}".format(obj))
        return None


def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]


class Progbar(object):
    def __init__(self, target, width=30, verbose=1, interval=0.01):
        '''Dislays a progress bar.

        # Arguments:
            target: Total number of steps expected.
            interval: Minimum visual progress update interval (in seconds).
        '''
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], force=False):
        '''Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
    return identifier
