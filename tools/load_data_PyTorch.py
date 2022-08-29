# -*- encoding: utf-8 -*-
"""
    @File        : load_data_PytORCH.py
    @Time        : 2022/8/18 11:53
    @Author      : Mirrich Wang 
    @Version     : Python 3.X.X (Anaconda)
    @Description : 仿 torchvision.ImageFolder 的读取图片数据
"""

import re
import os
import pandas as pd
import numpy as np
from PIL import Image

num_channels = 1


def ImageFolder(root, subset="train"):
    root = os.path.join(root, subset)
    label_names = pd.read_csv(root, os.path.dirname(root) + '.names')[0].tolist()
    images, labels = [], []
    for i, label in enumerate(label_names):
        filenames = os.listdir(os.path.join(root, label))
        for filename in filenames:
            img = Image.open(os.path.join(root, label, filename)).convert('L' if num_channels == 1 else "RGB")
            images.append(np.expand_dims(np.array(img), 0))
            labels.append(i)
    return np.concatenate(images), np.array(labels)


if __name__ == '__main__':
    # X_train, y_train = ImageFolder("../datasets/CIFAR10")
    # X_test, y_test = ImageFolder("../datasets/MNIST", "test")
    #
    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)
    subset = 'test'
    root = '../datasets/'
    name = 'CIFAR100'
    total = 0
    label_names = pd.read_csv(os.path.join(root, name, name + '.names'), header=None)[0].to_list()
    label_names = list(map(lambda i: re.sub(r'[\\/\s]', '·', i), label_names))
    print(label_names)
    for label in label_names:
        total += len(os.listdir(os.path.join(root, name, "test", str(label))))
    print(total)
