from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import re
import six
import time
import copy
import math
import glob
import shutil
import hashlib
import warnings
import cProfile
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


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def int2tuple(x, reps=1):
    if isinstance(x, int):
        return (x,) * reps
    else:
        return x


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


def retrieve_all_files(path):
    """Retrieve all files in a directory and all its sub-directories, sort file names ascendingly.
    # Arguments:
        path: the input directory.
    # Returns:
        A sorted list of all retrieved files.
    """
    return sorted([f for f in glob.glob(os.path.join(path, "**"), recursive=True) if os.path.isfile(f)])


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def copyfile(src, dst):
    """Copy a source file (`src`) to another place (`dst`), create directories if not exist.
    # Arguments:
        src: the source/input file.
        dst: the destination/output file.
    """
    path = os.path.dirname(dst)
    makedirs(path)  # create directories recursively
    shutil.copyfile(src, dst)


def copydir(src, dst):
    shutil.copytree(src, dst)


def md5sum(file_path):
    return hashlib.md5(open(file_path, 'rb').read()).hexdigest()


def totuple(x, reps=1):
    if (not isinstance(x, tuple)) and (not isinstance(x, list)):
        return (x,) * reps
    else:
        return x


def dist2p(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def compute_angle(u, v, k):
    '''
    Compute angle between two vectors: uv and uk
    :param u:
    :param v:
    :param k:
    :return:
    '''
    uv = (v[0] - u[0], v[1] - u[1])
    uk = (k[0] - u[0], k[1] - u[1])
    uv_length = np.sqrt(uv[0] ** 2 + uv[1] ** 2)
    uk_length = np.sqrt(uk[0] ** 2 + uk[1] ** 2)
    return np.arccos((uv[0] * uk[0] + uv[1] * uk[1]) / (uv_length * uk_length)) * 180 / np.pi


def str2npy(s):
    while True:
        m = re.search('[0-9|\.] +[0-9|\.]', s)
        if m is None:
            break
        s = s.replace(s[m.span()[0]:m.span()[1]],
                      s[m.span()[0]] + ',' + s[m.span()[1] - 1])
    s = re.sub(' *\n *', ',', s)
    return eval('np.array({})'.format(s))


def profileit(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        result = prof.runcall(func, *args, **kwargs)
        prof_file = os.path.join("./debug", func.__module__ + "." + func.__name__)
        prof.dump_stats(prof_file + '.dump')

        with open(prof_file + '.profile', 'w') as stream:
            stats = pstats.Stats(prof_file + '.dump', stream=stream)
            stats.sort_stats("cumtime")
            stats.print_stats()
        return result

    return wrapper


def parse_date_string_to_timestamp(text):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y', '%m/%d/%Y',
                '%Y/%m/%d %H:%M:%S', '%Y:%m:%d %H:%M:%S', '%Y:%m:%d %H:%M:%S.%f',
                '%m/%Y'):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    raise ValueError('no valid date format found for {}'.format(text))


def has_substr_in_list(s, str_list):
    for t in str_list:
        if t in s:
            return True
    return False
