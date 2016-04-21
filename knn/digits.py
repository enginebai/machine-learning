#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'enginebai'

import os

from numpy import *

from knn import knn_classify


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


def digit_classifier():
    # training phase
    digit_labels = []
    train_file_list = os.listdir('trainingDigits')
    train_file_num = len(train_file_list)
    train_vector = zeros((train_file_num, 1024))
    for i in range(train_file_num):
        # 檔案全名，包含副檔名
        file_full_name = train_file_list[i]
        # 純檔案名稱，不含副檔名
        file_name = file_full_name.split('.')[0]
        # 檔案名稱格式是"數字_計數.txt"，這行是要取得數字部分
        digit_name = int(file_name.split('_')[0])
        digit_labels.append(digit_name)
        train_vector[i, :] = img2vector('trainingDigits/%s' % file_full_name)

    # test phase
    test_file_list = os.listdir('testDigits')
    error_count = 0.0
    test_file_num = len(test_file_list)
    for i in range(test_file_num):
        file_full_name = test_file_list[i]
        file_name = file_full_name.split('.')[0]
        digit_name = int(file_name.split('_')[0])
        test_vector = img2vector('testDigits/%s' % file_full_name)
        classify_result = knn_classify(test_vector, train_vector, digit_labels, 3)
        print('Classify result = %s, real answer = %s' % (classify_result, digit_name))
        if classify_result != digit_name:
            error_count += 1

    print('Total error = %d, error rate = %f' % (error_count, (error_count / float(test_file_num))))
