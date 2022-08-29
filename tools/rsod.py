# -*- encoding: utf-8 -*-
"""
@File        : rsod.py
@Time        : 2022/8/1 14:56
@Author      : Mirrich Wang 
@Version     : Python 3.9.12 (Conda)
@Description : None
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import xyxy2xywh


import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

"""++++++++++++++++++++
@@@ Settings
+++++++++++++++++++++++"""

train_size = .8
val_size = .1
root = r'..\datasets\RSOD'
splits = ['train', 'val', 'test']
label_names = pd.read_csv(os.path.join(root, 'RSOD.names'), header=None)[0].to_list()
class_to_ind = dict(zip(label_names, range(len(label_names))))
filenames, txt_filenames = list(), list()
for label in label_names:
    filenames.extend(list(map(lambda i: os.path.splitext(i)[0], os.listdir(os.path.join(root, "raw", label, 'JPEGImages')))))
    txt_filenames = list(map(lambda i: os.path.splitext(i)[0], os.listdir(os.path.join(root, "raw", label, 'Annotation', 'labels'))))
dis_filenames = list(set(filenames) - set(txt_filenames))
# 清除不存在标签文件的文件名字
for i in dis_filenames:
    filenames.remove(i)

"""++++++++++++++++++++
@@@ 数据预处理
+++++++++++++++++++++++"""

# for split in splits:
#     os.makedirs(os.path.join(save_path, 'labels', split))
#     os.makedirs(os.path.join(save_path, 'images', split))
#
# train_size = int(train_size * len(filenames))
# val_size = int(val_size * len(filenames))
# print('Train: %s \t Val: %s\tTest: %s' % (train_size, val_size, len(filenames) - train_size - val_size))
#
# indices = np.arange(len(filenames))
# np.random.seed(42)
# np.random.shuffle(indices)
# train_indices = indices[:train_size]
# val_indices = indices[train_size: train_size + val_size]
# test_indices = indices[train_size + val_size:]
# filenames_ = np.array(filenames)
# filenames_ = dict(zip(splits, [filenames_[train_indices], filenames_[val_indices], filenames_[test_indices]]))
#
# for split in splits:
#     for i, filename in enumerate(filenames_[split]):
#         img = Image.open(os.path.join(raw_path, 'JPEGImages', filename + '.jpg'))
#         w, h = img.size
#         anno = pd.read_csv(os.path.join(raw_path, 'Annotation/labels', filename + '.txt'), sep='\t', header=None)
#         labels = list(map(lambda i: name_to_label[i], anno.values[:, 1]))
#         bboxes = anno.values[:, 2:]
#         new_bboxes = pd.DataFrame([[label, *xyxy2xywh(w, h, *bbox)] for label, bbox in zip(labels, bboxes)])
#         new_bboxes.to_csv(os.path.join(save_path, 'labels', split, f'{filename}.txt'), sep='\t', index=False, header=False)
#         img.save(os.path.join(save_path, 'images', split, filename + '.jpg'))
#         print(f'{split.title()} {i + 1}/{len(filenames_[split])} Saved...')

for split in splits:
    os.makedirs(os.path.join(root, "annotations", split), exist_ok=True)
    files = os.listdir(os.path.join(root, "images", split))
    filenames = list(map(lambda i: os.path.splitext(i)[0], files))
    label_names = list(map(lambda i: i.split("_")[0], filenames))
    for filename, label in tqdm(zip(filenames, label_names), total=len(filenames), desc=split):
        xml_file = open(os.path.join(root, "raw", label, "Annotation", "xml", filename + ".xml"), "r")
        with open(os.path.join(root, "annotations", split, filename + ".xml"), "w+") as f:
            f.write(xml_file.read())
