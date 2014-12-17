#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'enginebai'

from math import log

def calculate_entropy(dataset):
    """
    計算一個數據的entropy。
    :param dataset:
    :return:
    """
    num_entries = len(dataset)
    # 為所有可能的分類建立dict方便做查詢
    label_count = {}
    for feature_vector in dataset:
        current_label = feature_vector[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1

    entropy = 0.0
    for label in label_count:
        prob = float(label_count[label]) / num_entries
        entropy -= prob * log(prob, 2)
    return entropy

def create_data_set():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'yes'], [0, 1, 'yes'], [0, 1, 'y']]
    labels = ['no', 'flippers']
    return data_set, labels

def split_data_set(data_set, axis, value):
    new_data_set = []
    for feature_vector in data_set:
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            new_data_set.append(reduced_feature_vector)
    return new_data_set