#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'enginebai'

from numpy import *

def img2vector(file_name):
    """
    將圖像讀入後轉成向量。
    :param file_name:
    :return:
    """
    # 宣告 1 x 1024的numpy向量
    vector = zeros((1, 1024))
    with open(file_name, 'r') as f:
        # 依序讀入檔案的前32行
        for i in range(32):
            line = f.readline()
            # 將每行的頭32個字元儲存在向量
            for j in range(32):
                vector[0, 32 * i + j] = int(line[j])
    return vector

