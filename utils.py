import sys
import logging
import csv
from pathlib import Path
import matplotlib.pyplot as plt


import imageio
import tifffile

import torch
import torch.nn as nn
import numpy as np

from itertools import islice
import csv

def make_tuple(x):              #判断数据类型
    if isinstance(x, int):      #isinstance来判断一个对象是否是一个已知的类型，比如x是否为int.
        return x, x
    if isinstance(x, list) and len(x) == 1:
        return x[0], x[0]
    return x


class AverageMeter(object):      #损失
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val                      #当前接收值
        self.sum += val * n                 #总共的损失
        self.count += n                     #有多少个样本
        self.avg = self.sum / self.count    #返回的平均值


def get_logger(type,logpath=None):                #Console日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if logpath is not None:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        if type =="train":
            #输出到文件
            logging.basicConfig(
                                level=logging.INFO ,
                                # format='%(asctime)s %(filename)s[line:%(lineno)d]  %(levelname)s %(message)s',
                                # datefmt='%a, %d %b %Y %H:%M:%S',
                                filename='out_train.log',
                                filemode='w')
        elif type == "test":
            logging.basicConfig(
                level=logging.INFO,
                # format='%(asctime)s %(filename)s[line:%(lineno)d]  %(levelname)s %(message)s',
                # datefmt='%a, %d %b %Y %H:%M:%S',
                filename='out_test.log',
                filemode='w')
        logger.setLevel(logging.INFO)
        # logging.getLogger('matplotlib').setLevel(logging.WARNING)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger


def save_checkpoint(model, optimizer, path):        #保存训练模型参数
    if path.exists():
        path.unlink()
    model = model.module if isinstance(model, nn.DataParallel) else model
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state = {'state_dict': model.state_dict(),
                 'optim_dict': optimizer.state_dict()}
    if isinstance(path, Path):
        torch.save(state, str(path.resolve()))
    else:
        torch.save(state, str(path.resolve()))


def load_checkpoint(checkpoint, model, optimizer=None, map_location=None):      #加载模型参数
    if not checkpoint.exists():
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    state = torch.load(checkpoint, map_location=map_location)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(state['state_dict'])

    if optimizer:
        optimizer.load_state_dict(state['optim_dict'])
    return state


def log_csv(filepath, values, header=None, multirows=False):        #保存日志的一些信息，csv是一个表格
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)


def load_pretrained(model, pretrained, requires_grad=False):        #更换某一部分网络的参数
    if isinstance(model, nn.DataParallel):
        model = model.module
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained)['state_dict']
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False


def save_array_as_tif(matrix, path, profile=None, prototype=None):      #将数组改为tif图片
    # assert matrix.ndim == 2 or matrix.ndim == 3
    # if prototype:
    #     with rasterio.open(str(prototype)) as src:
    #         profile = src.profile
    #         # profile.update(dtype='int16')
    # with rasterio.open(path, mode='w', **profile) as dst:
    #     if matrix.ndim == 3:
    #         for i in range(matrix.shape[0]):
    #             dst.write(matrix[i], i + 1)
    #     else:
    #         dst.write(matrix, 1)
    np_transpose = np.ascontiguousarray(matrix.transpose((1,2,0)))
    # tifffile.imsave(path, np_transpose[:, :, 0:4])
    imageio.imwrite(path,np_transpose[:, :, 0:4])

#打印损失函数pdf
def plot_loss(train,val, type, epoch):
    axis = np.linspace(1, epoch, epoch)
    # label = '{} '.format(type)
    # plt.title(label)
    fig = plt.figure()
    plt.plot(axis,train,label='train')
    plt.plot(axis,val,label='val')
    plt.legend(loc = 'best')
    plt.xlabel('Epochs')
    plt.ylabel(type)
    y_ticks = np.arange(0,150,10)
    x_ticks = np.arange(0,30,2)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)

    plt.grid(True)
    plt.savefig('./out/mertics/{}.jpg'.format(type))
    plt.close(fig)

def CSV_plot():
    train_metric = {
        'loss': [],
        'mse': [],
        'psnr': [],
        'ssim': [],
        'sam': [],
        'ergas': [],
        'cc': [],
        'rmse': [],
    }
    val_metric = {
        'loss': [],
        'mse': [],
        'psnr': [],
        'ssim': [],
        'sam': [],
        'ergas': [],
        'cc': [],
        'rmse': [],

    }
    csv_reader = csv.reader(open("./out/train/history.csv"))  # 从CSV中读取的数据都是字符串
    total = 0
    for row in islice(csv_reader, 1, None):
        if (len(row) != 0):
            total = total + 1
            train_metric['loss'].append(float(row[1]))
            train_metric['mse'].append(float(row[2]))
            val_metric['loss'].append(float(row[3]))
            val_metric['mse'].append(float(row[4]))
            plot_loss(train_metric['loss'],val_metric['loss'], 'loss', total)
            plot_loss(train_metric['mse'],val_metric['mse'], 'mse', total)
