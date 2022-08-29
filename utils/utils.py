# -*- encoding: utf-8 -*-
"""
    @File        : utils.py
    @Time        : 2022/8/16 15:03
    @Author      : Mirrich Wang 
    @Version     : Python 3.9.12 (ENV)
    @Description : None
"""

import os

import pickle
import unzip
import struct
import xml.dom.minidom

import numpy as np

from PIL import Image
from tqdm import tqdm


def load_meta(file):
    """
    读取 meta 格式的数据，
    :param file: [str] meta 文件路径
    :return: [dict] 读取的字典
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='iso-8859-1')
    return data


def load_byte(file, cache='>IIII', dty=np.uint8):
    """
    读取 ubyte 格式的文件
    :param file: [str] ubyte 文件路径
    :param cache: [str] ubyte 标记符号
    :param dty: [type] 读取后转换的类型 Default -> numpy.uint8
    :return: data [np.array] 读取的 numpy 矩阵数据
    """
    iter_num = cache.count('I') * 4
    with open(file, 'rb') as f:
        magic = struct.unpack(cache, f.read(iter_num))
        data = np.fromfile(f, dtype=dty)
    return data


def array_to_images(root, data, subsets=None):
    """ -< 用于 Image Classification 数据集 >-
    矩阵转换成图片并保存
    :param root: [str] 图片保存路径
    :param data: [dict] 存储对应文件名字、图片矩阵、标签、标签文字字符串排序列表的字典
                        {"filenames": ..., "data": ..., "labels": ... "label_names": ...}
    :param subsets: [list[str]][optional] 划分子集，可输入"train", "valid", "test" Default -> None
    :return: "Done"
    """
    if subsets is not None:
        names = data['label_names']
        for subset in subsets:
            size = len(data[subset]['filenames'])
            filenames = data[subset]['filenames']
            labels = data[subset]['labels']
            images = data[subset]['data']
            iters = zip(filenames, images, labels)
            for filename, img_numpy, label in tqdm(iters, desc=root + rf'\{subset}', total=size):
                save_dir = os.path.join(root, subset, names[label], filename)
                if not os.path.exists(os.path.dirname(save_dir)):
                    os.makedirs(os.path.dirname(save_dir))
                if img_numpy.shape[2] == 1:
                    img_numpy = np.concatenate([img_numpy] * 3, axis=2)
                img = Image.fromarray(img_numpy)
                img.save(save_dir)
    else:
        size = len(data['filenames'])
        names = data['label_names']
        filenames = data['filenames']
        labels = data['labels']
        images = data['data']
        for filename, img_numpy, label in tqdm(zip(filenames, images, labels), desc=root, total=size):
            h, w, c = img_numpy.shape
            save_dir = os.path.join(root, names[label], filename)
            if not os.path.exists(os.path.dirname(save_dir)):
                os.makedirs(os.path.dirname(save_dir))
            img = Image.fromarray(img_numpy)
            img.save(save_dir)
    return 'done'


def xyxy2xywh(w, h, x1, y1, x2, y2):
    """
    [xmin, ymin, xmax, ymax] 转换成YOLO-V5格式的 [x, y, w, h]
    :param w: [int] 图片宽度 height
    :param h: [int] 图片高度 width
    :param x1: [float] 左上角横坐标
    :param y1: [float] 左上角纵坐标
    :param x2: [float] 右下角横坐标
    :param y2: [float] 右下角纵坐标
    :return: [list] bbox 信息
    """
    x = ((x1 + x2) / 2) / w  # x center
    y = ((y1 + y2) / 2) / h  # y center
    w = (x2 - x1) / w  # width
    h = (y2 - y1) / h  # height
    try:
        x = np.where(x > 0, x, 0)
        x = np.where(x < 1, x, 1)
        y = np.where(y > 0, y, 0)
        y = np.where(y < 1, y, 1)
        w = np.where(w > 0, w, 0)
        w = np.where(w < 1, w, 1)
        h = np.where(h > 0, h, 0)
        h = np.where(h < 1, h, 1)
    except:
        pass
    return [float(x), float(y), float(w), float(h)]


def xml2txt(file, label_names):
    """
    xml标注转换成YOLO-V5格式
    :param file: [str] xml 文件
    :param label_names: [list[str]] 标签列表
    :return: [list] YOLO-V5格式的 bbox
    """
    # 打开xml文档
    dom = xml.dom.minidom.parse(file)
    # 得到文档元素对象
    root = dom.documentElement
    name = root.getElementsByTagName('filename')[0].firstChild.data.split(".")[0]
    w = root.getElementsByTagName('width')[0].firstChild.data.split(".")[0]
    h = root.getElementsByTagName('height')[0].firstChild.data.split(".")[0]

    class_to_ind = dict(zip(label_names, range(len(label_names))))

    data = root.getElementsByTagName('object')
    new = []
    for doc in data:
        sens = doc.getElementsByTagName('name')[0].firstChild.data
        xmin = doc.getElementsByTagName('xmin')[0].firstChild.data
        ymin = doc.getElementsByTagName('ymin')[0].firstChild.data
        xmax = doc.getElementsByTagName('xmax')[0].firstChild.data
        ymax = doc.getElementsByTagName('ymax')[0].firstChild.data
        xywh = xyxy2xywh(float(w), float(h), float(xmin), float(ymin), float(xmax), float(ymax))
        new.append([class_to_ind[sens], xywh[0], xywh[1], xywh[2], xywh[3]])
    return new


if __name__ == '__main__':
    path = '../datasets/CIFAR10/raw/cifar-10-batches-py/data_batch_1'
    print(load_meta(path).keys())
