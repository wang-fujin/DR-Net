
import os
import colorlog
import logging
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Scaler():
    def __init__(self,data1,data2=None):  # data.shape (N,L,C)  或者 (N,C)
        if data2 is None:
            self.data = data1
        else:
            self.train_num = data1.shape[0]
            self.data = np.concatenate((data1, data2), axis=0)
        self.data2 = data2
        if self.data.ndim == 3:
            self.mean = self.data.mean(axis=(0,1)).reshape(1,1,-1)
            self.var = self.data.var(axis=(0,1)).reshape(1,1,-1)
            self.max = self.data.max(axis=(0,1)).reshape(1,1,-1)
            self.min = self.data.min(axis=(0,1)).reshape(1,1,-1)
        elif self.data.ndim ==2:
            self.mean = self.data.mean(axis=0).reshape(1, -1)
            self.var = self.data.var(axis=0).reshape(1, -1)
            self.max = self.data.max(axis=0).reshape(1, -1)
            self.min = self.data.min(axis=0).reshape(1, -1)
        elif self.data.ndim == 1: # label data
            self.mean = self.data.mean()
            self.var = self.data.var()
            self.max = self.data.max()
            self.min = self.data.min()
        else:
            raise ValueError('data dim error!')

    def standerd(self):
        X = (self.data - self.mean) / self.var
        if self.data2 is None:
            return X
        else:
            train = X[:self.train_num]
            test = X[self.train_num:]
            return train, test

    def minmax(self):
        X = (self.data - self.min) / (self.max - self.min)
        if self.data2 is None:
            return X
        else:
            train = X[:self.train_num]
            test = X[self.train_num:]
            return train, test


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_logger(filename=None,con_level='debug',file_level='debug'):
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    level_dict = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'ctirical': logging.CRITICAL
    }

    logger1 = logging.getLogger('my_log')
    logger1.setLevel(logging.DEBUG)

    consoleHander = logging.StreamHandler()
    consoleHander.setLevel(level_dict.get(con_level))

    if filename is not None:
        fileHander = logging.FileHandler(filename)
        fileHander.setLevel(level_dict.get(file_level))

    formatter1 = logging.Formatter("[%(asctime)s.%(msecs)03d] %(threadName)s %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s")
    formatter2 = colorlog.ColoredFormatter(fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(threadName)s %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
                                           log_colors=log_colors_config)

    consoleHander.setFormatter(formatter2)
    if filename is not None:
        fileHander.setFormatter(formatter1)

    logger1.addHandler(consoleHander)
    if filename is not None:
        logger1.addHandler(fileHander)
        return logger1, consoleHander, fileHander
    else:
        return logger1, consoleHander



