# -*- encoding: utf-8 -*-
"""
@File        : utils/cifar100.py
@Time        : 2022/7/31 14:01
@Author      : Mirrich Wang 
@Version     : Python 3.9.12 (Conda)
@Description : None
"""

# 包函数
from __init__ import unpickle, save_images
# 文件操作
import os
# 数据处理
import numpy as np
import pandas as pd
# 数据集下载
from torchvision.datasets import CIFAR100
# 进程池
from concurrent.futures import ThreadPoolExecutor

"""+++++++++++++++++++++++
@@@ Settings
++++++++++++++++++++++++++"""

image_size = 32
num_channels = 3
path = '../data/CIFAR100'
splits = ['train', 'test']
cifar100 = CIFAR100(root=path + '/raw', download=True)

"""+++++++++++++++++++++
@@@ 数据预处理
++++++++++++++++++++++++"""

# 读取标签 meta 文件的数据
label_info = unpickle(os.path.join(path, 'raw', 'cifar-100-python', 'meta'))
label_names = label_info['fine_label_names']  # 小类
coarse_names = label_info['coarse_label_names']  # 大类

# 读取图片 meta 文件
train_raw = unpickle(os.path.join(path, 'raw', 'cifar-100-python', 'train'))
test_raw = unpickle(os.path.join(path, 'raw', 'cifar-100-python', 'test'))

# 将图片进行转换，打包数据
train, test = {}, {}
train['images'] = train_raw['data'].reshape(-1, num_channels, image_size, image_size).transpose(0, 2, 3, 1)
train['labels'] = np.array(train_raw['fine_labels'])
train['filenames'] = train_raw['filenames']

test['images'] = test_raw['data'].reshape(-1, num_channels, image_size, image_size).transpose(0, 2, 3, 1)
test['labels'] = np.array(test_raw['fine_labels'])
test['filenames'] = test_raw['filenames']
data = dict(zip(splits, [train, test]))

print('Train: Images - %s\tLabels - %s' % (train['images'].shape, train['labels'].shape))
print('Test: Images - %s\tLabels - %s' % (test['images'].shape, test['labels'].shape))
print('Label names (Torch) :', cifar100.classes)

"""++++++++++++++++++++
@@@ 数据预处理
+++++++++++++++++++++++"""

coarse_fine = dict(zip(coarse_names, [''] * len(coarse_names)))
coarse_fine_raw = set(zip(train_raw['coarse_labels'], train_raw['fine_labels']))
for i in coarse_fine_raw:
    coarse_fine[coarse_names[i[0]]] += str(label_names[i[1]] + ', ')
coarse_fine = dict(map(lambda i: (i[0], i[1].rstrip(', ')), coarse_fine.items()))
pd.DataFrame(label_names).to_csv(os.path.join(path, 'CIFAR100.names'), index=False, header=False)
pd.DataFrame(coarse_fine.items()).to_csv(os.path.join(path, 'CIFAR100.coarse.csv'), index=False, header=False)
for split in splits:
    for i in label_names:
        os.makedirs(os.path.join(path, split, i))

"""+++++++++++++++++++++
@@@ 图片存储
++++++++++++++++++++++++"""

executor = ThreadPoolExecutor(max_workers=2)
args = [path, data, label_names]

pool = ThreadPoolExecutor(max_workers=2)
future1 = pool.submit(save_images, *args+['train'])
future2 = pool.submit(save_images, *args+['test'])

print('Train', future1.result())
print('Test', future2.result())
