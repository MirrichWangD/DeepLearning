# -*- coding: utf-8 -*-
"""
    @ Author        ：Mirrich Wang
    @ Created       ：2022/7/16 16:23
    @ Description   ：None
"""

import os
import pandas as pd

from PIL import Image
from tqdm import tqdm
from utils import xml2txt

"""++++++++++++++++++
@@@ Settings
+++++++++++++++++++++"""

years = ["2007", "2012"]
splits = ["train", "val", "test"]
root = r"..\datasets\VOC"
devkit_path = os.path.join(root, "raw", "VOCdevkit")
label_names = pd.read_table(os.path.join(root, "VOC.names"), header=None)[0].to_list()

"""++++++++++++++++++
@@@ 数据预处理
+++++++++++++++++++++"""


def VOCDetection():
    # 获取年份 year
    for year in years:
        anno_path = os.path.join(devkit_path, "VOC" + year, "Annotations")  # 获取Annotation目录
        img_path = os.path.join(devkit_path, "VOC" + year, "JPEGImages")  # 获取图片目录
        # 获取划分集文件名称 split (train, val, test)
        for split in splits:
            set_file = os.path.join(devkit_path, "VOC" + year, r"ImageSets\Main", split + ".txt")
            image_set = pd.read_table(set_file, header=None, dtype=str)[0]

            # 获得图片和标签保存地址，并且创建文件夹
            save_img_path = os.path.join(root, "images", split + year)
            save_anno_path = os.path.join(root, "labels", split + year)
            os.makedirs(save_img_path, exist_ok=True)
            os.makedirs(save_anno_path, exist_ok=True)

            # 获取图片和标注文件名字
            for img_name in tqdm(image_set, total=len(image_set), desc=split + year):
                img = Image.open(os.path.join(img_path, img_name + ".jpg"))  # 读取图片
                img.save(os.path.join(save_img_path, img_name + ".jpg"))

                # VOC2012 测试图片没有 annotations，因此需要跳过
                if (split, year) != ("test", "2012"):
                    # 读取 xml 文件并且转换成 YOLO-V5 数据格式
                    xml_file = os.path.join(anno_path, img_name + ".xml")
                    txt_file = os.path.join(save_anno_path, img_name + ".txt")
                    bbox = pd.DataFrame(xml2txt(xml_file, label_names))
                    bbox.to_csv(txt_file, index=False, header=False, sep=" ")


if __name__ == "__main__":
    print(os.getcwd())
    # VOCDetection()
