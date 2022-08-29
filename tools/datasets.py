# -*- encoding: utf-8 -*-
"""
    @File        : datasets.py
    @Time        : 2022/8/16 19:02
    @Author      : Mirrich Wang 
    @Version     : Python 3.8.8 (Anaconda)
    @Description : 自定义构造的数据集相关类，可以进行下载、转存图片的操作
"""

import re
import os
import sys

sys.path.append('../')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import ceil
from utils import load_byte, load_meta, array_to_images


class Mnist:
    """ 自定义 MNIST 数据集相关类 """

    name = "MNIST"  # 数据集名字
    num_channels = 1  # 图片通道数
    image_size = [28, 28]  # 图片尺寸
    label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # 标签名字

    base_folder = ""  # 存放原始文件目录

    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    def __init__(self, root='.', save_path=None, download=False, vis_sample=None):
        """
        Mnist 类构造方法
        :param root: [str] 数据集路径
        :param save_path: [str] 图片转存路径，输入后将自动保存图片到本地
        :param download: [bool] 是否需要下载原始文件
        :param vis_sample: [bool] 是否展示一定数量的图片
        """
        self.root = root
        self.raw_dir = os.path.join(self.root, "raw", self.base_folder)  # 存放原始文件路径

        # 是否需要下载数据集源文件
        if download:
            self.download()

        # 获取数据信息
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = self.load_data()
        self.train_size, self.test_size = len(self.train_data), len(self.test_data)

        print(self.__repr__())

        # 是否需要展示部分数据集图片
        if vis_sample:
            self.plot_sample()

        if save_path:
            self.save_data(save_path)

    def __repr__(self):
        """ 返回相关字符串信息 """
        train_rate = self.train_size / (self.train_size + self.test_size)
        test_rate = self.test_size / (self.train_size + self.test_size)
        info = {
            "Dataset Name": "%s" % self.name,
            "Num channels": "%i" % self.num_channels,
            "Image size": "%s" % self.image_size,
            "Train size": "%i (%.2f%%)" % (self.train_size, train_rate * 100),
            "Test size": "%i (%.2f%%)" % (self.test_size, test_rate * 100),
            "Labels": "%s" % len(self.label_names),
        }
        s = [" Info ".center(50, "="), "\n".join(['| %-21s | %-22s |' % (k, v) for k, v in info.items()]), "=" * 50]
        return "\n".join(s)

    def load_data(self):
        """
        读取源文件，进行预处理得到图片矩阵和标签信息
        :return: (训练集图片矩阵, 训练集标签矩阵), (测试集图片举证, 测试集标签矩阵)
        """
        data = {}
        for subset in ['train', 'test']:
            subset_ = "t10k" if subset == "test" else subset
            images = load_byte(os.path.join(self.raw_dir, f'{subset_}-images-idx3-ubyte'))
            labels = load_byte(os.path.join(self.raw_dir, f'{subset_}-labels-idx1-ubyte'), '>II')
            images = images.reshape([-1, self.num_channels] + self.image_size).transpose(0, 2, 3, 1)
            data[subset] = (images, labels)
        return data.values()

    def plot_sample(self, num=32):
        """ 展示部分案例图片，默认 100 张 """
        # 随机抽取32张图片
        ids = np.random.randint(0, self.test_size, num)

        images, labels = self.test_data[ids], self.test_labels[ids]

        fig, ax = plt.subplots(ceil(num / 8), 8, tight_layout=True)  # 调整分图数量
        for i, (image, label) in enumerate(zip(images, labels)):
            r, c = int(i / 8), i % 8

            ax[r][c].imshow(image, 'gray' if self.num_channels == 1 else None)
            ax[r][c].set_title(self.label_names[label], fontsize=10)  # 显示标签名字在图片上方
            # 关闭坐标轴
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
        plt.show()

    def save_data(self, path):
        """ 将图片保存到 path 中 """
        data = {"train": {"data": self.train_data,
                          "labels": self.train_labels},
                "test": {"data": self.test_data,
                         "labels": self.test_labels},
                "label_names": list(map(lambda i: re.sub(r'[\\/:*?"<>|\s]', "·", i), self.label_names))}
        data["train"]['filenames'], data["test"]["filenames"] = self.get_filenames()
        print(len(data["train"]['filenames']), len(data['test']['filenames']))
        print(f'Convert np.array -> PIL.Image and save to {path}...')
        array_to_images(path, data, ['train', 'test'])
        print('Save Completed!')

    def get_filenames(self):
        filenames = {"train": [], "test": []}
        idx = dict(zip(range(len(self.label_names)), [{"train": 0, "test": 0}] * len(self.label_names)))
        for subset in ["train", "test"]:
            labels = load_byte(os.path.join(self.raw_dir, f'{subset.replace("est", "10k")}-labels-idx1-ubyte'), '>II')
            for i, label in enumerate(labels):
                label_name = re.sub(r'[\\/:*?"<>|\s]', "·", self.label_names[label])
                filenames[subset].append("%s_%s_%i.png" % (subset, label_name, idx[label][subset]))
                idx[label][subset] += 1
        return filenames.values()

    def download(self):
        from torchvision.datasets import MNIST
        MNIST(root=self.root.replace(r'\MNIST', ''), download=True)

    # def _check_integrity(self) -> bool:
    #     root = self.raw_dir
    #     for filename, md5 in resources:
    #         fpath = os.path.join(self.raw_dir, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True
    #
    # def download(self) -> None:
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

class FashionMnist(Mnist):
    """ 自定义 FashionMNIST 数据集相关类 """

    name = "FashionMNIST"
    label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']  # 标签名字

    def download(self):
        from torchvision.datasets import FashionMNIST
        FashionMNIST(root=self.root.replace(r'\FashionMNIST', ''), download=True)


class Cifar10(Mnist):
    """ CIFAR10 自定义数据集类，继承 MNIST的方法 """

    name = "CIFAR10"
    num_channels = 3
    image_size = [32, 32]
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    base_folder = "cifar-10-batches-py"

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def load_data(self):
        """ 读取 meta 文件获得数据矩阵 """
        data = {}
        for subset in ["train", "test"]:
            meta = list(map(lambda i: load_meta(os.path.join(self.raw_dir, i[0])), eval(f'self.{subset}_list')))
            images = np.concatenate(list(map(lambda i: i['data'], meta)))
            images = images.reshape([-1, self.num_channels] + self.image_size).transpose(0, 2, 3, 1)
            labels = np.concatenate(list(map(lambda i: i[self.meta['key'].replace("_names", "s")], meta)))
            data[subset] = (images, labels)

        return zip(*zip(*data.values()))
        # train_meta = list(map(lambda i: load_meta(os.path.join(self.raw_dir, i[0])), self.train_list))
        # train_images = np.concatenate(list(map(lambda i: i['data'], train_meta)))
        # train_data = train_images.reshape([-1, self.num_channels] + self.image_size).transpose(0, 2, 3, 1)
        # train_labels = np.concatenate(list(map(lambda i: i[self.meta['key'].split('_')[0] + 's'], train_meta)))
        # # 测试集数据处理
        # test_meta = list(map(lambda i: load_meta(os.path.join(self.raw_dir, i[0])), self.test_list))
        # test_images = np.concatenate(list(map(lambda i: i['data'], test_meta)))
        # test_data = test_images.reshape([-1, self.num_channels] + self.image_size).transpose(0, 2, 3, 1)
        # test_labels = np.concatenate(list(map(lambda i: i[self.meta['key'].split('_')[0] + 's'], test_meta)))

        # return (train_data, train_labels), (test_data, test_labels)

    def get_filenames(self):
        train_meta = list(map(lambda i: load_meta(os.path.join(self.raw_dir, i[0])), self.train_list))
        train_filenames = np.concatenate(list(map(lambda i: i['filenames'], train_meta))).tolist()

        test_meta = list(map(lambda i: load_meta(os.path.join(self.raw_dir, i[0])), self.test_list))
        test_filenames = np.concatenate(list(map(lambda i: i['filenames'], test_meta))).tolist()

        return {"train": train_filenames, "test": test_filenames}

    def download(self):
        from torchvision.datasets import CIFAR10
        CIFAR10(root=self.raw_dir, download=True)


class Cifar100(Cifar10):
    """ 自定义 CIFAR100 数据集相关类，继承 Cifar10 类 """

    name = "CIFAR100"
    label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                   'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                   'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
                   'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
                   'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
                   'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
                   'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                   'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    base_folder = "cifar-100-python"

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def download(self):
        from torchvision.datasets import CIFAR100
        CIFAR100(root=self.raw_dir, download=True)


if __name__ == "__main__":
    mnist = Mnist(root=r'..\datasets\MNIST', vis_sample=True)
    mnist.save_data(r'..\datasets\MNIST')
    fashion_mnist = FashionMnist(root=r'..\datasets\FashionMNIST', vis_sample=True)
    cifar10 = Cifar10(root=r'..\datasets\CIFAR10', vis_sample=True)
    cifar100 = Cifar100(root=r'..\datasets\CIFAR100', vis_sample=True)
