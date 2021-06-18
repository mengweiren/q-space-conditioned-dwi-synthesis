"""This module contains simple helper functions """
from __future__ import print_function
import numpy as np
import os
from time import time
from tqdm import tqdm
import yaml

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def convert(seconds):
    '''convert time format
    :param seconds:
    :return:
    '''
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)

def MAE(pred, trg, mask):
    dif = np.abs(trg - pred)
    return np.sum(dif) / np.sum(mask)

def eval_4d(pred, trg, metric, mask=None):
    num_3d = pred.shape[-1]
    all_metrics = []
    for m in metric:
        st = time()
        all_ = list()
        w, h, z, d = pred.shape
        for i in tqdm(range(num_3d)):
            if m.__name__ == 'MAE':
                all_.append(MAE(pred[..., i], trg[..., i], mask))  # mask MAE
            else:
                all_.append(m(pred[..., i], trg[..., i]))
        all_metrics += [np.mean(all_), np.var(all_)]
        et = time()
        print('{} in {}'.format(m.__name__, convert(et - st)))
    return all_metrics
