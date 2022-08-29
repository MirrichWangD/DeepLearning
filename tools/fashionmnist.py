# -*- encoding: utf-8 -*-
"""
@File        : utils/fashionmnist.py
@Time        : 2022/8/1 9:10
@Author      : Mirrich Wang 
@Version     : Python 3.9.12 (Conda)
@Description : None
"""

from __init__ import load_byte, save_images
# 文件操作
import os
# 数据处理
import re
import numpy as np
import pandas as pd
# 数据集下载
from torchvision.datasets import FashionMNIST
# 进程池
from concurrent.futures import ThreadPoolExecutor

"""+++++++++++++++++++++++
@@@ Settings
++++++++++++++++++++++++++"""

image_size = 28  # 图像尺寸
filename_fmt = '%s_%i.png'
path = '../data/FashionMNIST'  # 数据路径
splits = ['train', 'test']  # 训练，测试集划分（原始数据已经划分）
fashionmnist = FashionMNIST(root=path[:-13], download=True)  # 设置路径，下载数据集
pd.DataFrame(fashionmnist.classes).to_csv(os.path.join(path, 'FashionMNIST.names'), index=False, header=False)
label_names = list(map(lambda i: re.sub('[/ ]', '_', i), fashionmnist.classes))

"""+++++++++++++++++++++++
@@@ 数据预处理
++++++++++++++++++++++++++"""

for split in splits:
    for label_name in label_names:
        os.makedirs(os.path.join(path, split, label_name))


# 去读图片和标签的 ubyte文件，转换图片维度
def load_mnist(root, kind='train'):
    images = load_byte(os.path.join(root, '%s-images-idx3-ubyte' % kind), '>IIII')
    labels = load_byte(os.path.join(root, '%s-labels-idx1-ubyte' % kind), '>II')
    images = images.reshape(-1, image_size, image_size)
    return images, labels


(X_train, y_train) = load_mnist(os.path.join(path, 'raw'), 'train')
(X_test, y_test) = load_mnist(os.path.join(path, 'raw'), 't10k')
# 处理 MNIST 数据集图片名字
train_ids = dict(zip(label_names, [0] * len(label_names)))
test_ids = train_ids.copy()
train_filenames, test_filenames = [], []
for label in y_train:
    train_filenames.append(filename_fmt % (label_names[label], train_ids[label_names[label]]))
    train_ids[label_names[label]] += 1
for label in y_test:
    test_filenames.append(filename_fmt % (label_names[label], test_ids[label_names[label]]))
    test_ids[label_names[label]] += 1

train, test = {}, {}
train['images'] = X_train
train['labels'] = y_train
train['filenames'] = train_filenames

test['images'] = X_test
test['labels'] = y_test
test['filenames'] = test_filenames
data = dict(zip(splits, [train, test]))

print('Train: Images - %s\tLabels - %s' % (train['images'].shape, train['labels'].shape))
print('Test: Images - %s\tLabels - %s' % (test['images'].shape, test['labels'].shape))
print('Label names (Torch) :', label_names)

"""+++++++++++++++++++++
@@@ 图片存储
++++++++++++++++++++++++"""

args = [path, data, label_names]

pool = ThreadPoolExecutor(max_workers=2)

future1 = pool.submit(save_images, *args + ['train'])
future2 = pool.submit(save_images, *args + ['test'])

print('Train', future1.result())
print('Test', future2.result())
