# -*- encoding: utf-8 -*-
"""
@File        : coco.py
@Time        : 2022/8/1 16:56
@Author      : Mirrich Wang 
@Version     : Python 3.9.12 (Conda)
@Description : None
"""

from __init__ import xyxy2xywh

import os

import json
import numpy as np
import pandas as pd

from PIL import Image

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

"""+++++++++++++++++++++++
@@@ Settings
++++++++++++++++++++++++++"""

years = ['2014', '2017']
splits = ['train', 'val']
raw_path = '../data/COCO/raw'
data_path = '../data/COCO'

"""++++++++++++++++++++++++
@@@ 数据预处理
+++++++++++++++++++++++++++"""


def load_json(root, kind, year):
    with open(os.path.join(root, 'instances_%s%s.json' % (kind, year)), 'r') as f:
        data = json.load(f)
    return data


def action(split, year):
    def get_data(i, image_id):
        # 获取图片名字、Bounding Box、标签列表
        filename = images[image_id][:-4]
        bbox = annotation[annotation['image_id'] == image_id]['bbox'].to_list()
        labels = annotation[annotation['image_id'] == image_id]['category_id'].to_list()
        # 读取图片，转换成 YOLOV5 格式
        img = Image.open(f'{raw_path}/{split}{year}/{filename}.jpg')
        W, H = img.size  # 获取图片宽度和长度
        new_bboxes = [[label, *xyxy2xywh(W, H, x, y, x + w, y + h)] for label, (x, y, w, h) in zip(labels, bbox)]
        new_bboxes = pd.DataFrame(new_bboxes)
        new_bboxes.to_csv(f'{data_path}/labels/{split}{year}/{filename}.txt', index=False, header=False, sep='\t')
        img.save(f'{data_path}/images/{split}{year}/{filename}.jpg')
        if i % 100 == 0:
            print(f'{split}{year} {i + 1}/{len(images)} Saved...')
        return 'done'

    os.makedirs(os.path.join(data_path, 'images', f'{split}{year}'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'labels', f'{split}{year}'), exist_ok=True)
    instances = load_json(f'{raw_path}/annotations', split, year)
    images = dict(map(lambda i: (i['id'], i['file_name']), instances['images']))
    annotation = pd.DataFrame(instances['annotations'])[['image_id', 'category_id', 'bbox']]
    categories = pd.DataFrame(instances['categories'])
    pool = ThreadPoolExecutor(max_workers=20)
    result = pool.map(get_data, range(len(images)), images.keys())
    return 'done'


if __name__ == "__main__":
    for split in splits:
        for year in years:
            action(split, year)
