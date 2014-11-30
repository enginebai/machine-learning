#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'enginebai'

from numpy import *
import operator

def create_data_set():
    """
    產生訓練集合。
    :return:
    """
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def knn_classify(input, data_set, labels, k):
    """
    執行kNN分類演算法
    :param input: 輸入新樣本
    :param data_set: 訓練集合樣本
    :param labels: 訓練集合分類
    :param k: k值
    :return: 新樣本分類
    """
    # calculate the distance (cosine similarity)
    data_set_size = data_set.shape[0]
    diff_mat = tile(input, (data_set_size, 1)) - data_set
    sqrt_diff_mat = diff_mat ** 2
    sqrt_distance = sqrt_diff_mat.sum(axis=1)
    distances = sqrt_distance ** 0.5
    sorted_sidtance_indicies = distances.argsort()

    # find k nearest neighbor
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_sidtance_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    # return the knn label
    return sorted_class_count[0][0]

def file2matrix(file_name):
    """
    將輸入檔案的數值轉成NumPy可以解析的物件格式。
    :param file_name:
    :return:
    """
    with open(file_name, 'r') as f:
        array_lines = f.readlines()
        number_lines = len(array_lines)
        num_matrix = zeros((number_lines,3))
        class_label_vector = []
        index = 0
        for line in array_lines:
            line = line.strip()
            list_from_line = line.split('\t')
            num_matrix[index, :] = list_from_line[0:3]
            label_text = list_from_line[-1]
            label = 0
            if label_text == 'didntLike':
                label = 1
            elif label_text == 'smallDoses':
                label = 2
            elif label_text == 'largeDoses':
                label = 3
            class_label_vector.append(label)
            index += 1
    return num_matrix, class_label_vector





