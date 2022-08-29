# -*- coding: utf-8 -*-
"""
    @Author        ：Mirrich Wang
    @Created       ：2022/4/25 13:49
    @Description   ：Preprocessing data
"""

import os
from sklearn.model_selection import train_test_split


def polarity_split(data_path, save_path, test_size=.1, seed=42):
    idx2label = {0: 'neg', 1: 'pos'}
    split_list = ['train', 'test']
    count = dict(zip(split_list, [dict(zip(range(2), [0] * 2))] * 2))
    texts, labels = [], []
    for i in range(len(idx2label)):
        label_files = os.listdir(os.path.join(data_path, idx2label[i]))
        label_texts = [open(os.path.join(data_path, idx2label[i], file)).read() for file in label_files]
        texts += label_texts
        labels += [i] * len(label_files)
        print('Load {} files:'.format(idx2label[i]), len(label_files))

    data = list(zip(texts, labels))
    alldata = dict(zip(split_list, train_test_split(data, test_size=test_size, random_state=seed)))
    for split in split_list:
        os.makedirs(os.path.join(save_path, split, 'pos'), exist_ok=True)
        os.makedirs(os.path.join(save_path, split, 'neg'), exist_ok=True)
        for text, label in alldata[split]:
            f = open(os.path.join(save_path, split, idx2label[label], f'{label}_{count[split][label]}.txt'), 'w+')
            f.write(text)
            f.close()
            count[split][label] += 1


if __name__ == '__main__':
    polarity_split('../datasets/Text Classification/ReviewPolarity/txt_sentoken',
                   '../datasets/Text Classification/ReviewPolarity')
