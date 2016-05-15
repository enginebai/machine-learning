#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import jieba

__author__ = 'enginebai'

jieba.load_userdict('data/dict_zhTW.txt')


def save_file(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)


def read_file(path):
    with open(path, 'r', encoding='utf8') as file:
        content = file.read()
    return content


def seg_corpus():
    # 語料庫和斷字後的路徑
    corpus_path = 'data/corpus/'
    seg_path = 'data/seg/'

    corpus_dirs = os.listdir(corpus_path)
    for dir in corpus_dirs:
        class_path = corpus_path + dir + '/'
        seg_dir = seg_path + dir + '/'
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        for file in os.listdir(class_path):
            print("Segment for", class_path + file)
            content = read_file(class_path + file).replace('\r\n','').strip()
            content_seg = jieba.cut(content)
            save_file(seg_dir + file, ' '.join(content_seg))


if __name__ == '__main__':
    seg_corpus()
