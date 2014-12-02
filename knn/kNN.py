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

def normalize(data_set):
    """
    Normalize the data set value to range(0, 1)
    :param data_set:
    :return:
    """
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    range_val = max_val - min_val
    normalize_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    normalize_data_set = data_set - tile(min_val, (m, 1))
    normalize_data_set = normalize_data_set / tile(range_val, (m, 1))
    return normalize_data_set, range_val, min_val

def knn_classifier_test():
    # 訓練集合資料用來當作測試的比例
    test_ratio = 0.05
    dating_data_matrix, dating_labels = file2matrix('datingTestSet.txt')
    normalize_dating_data_matrix, ranges, min_val = normalize(dating_data_matrix)

    print normalize_dating_data_matrix
    import matplotlib.pyplot as plot
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(normalize_dating_data_matrix[:, 0], normalize_dating_data_matrix[:, 2],
               15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plot.show()

    m = normalize_dating_data_matrix.shape[0]
    num_test = int(m * test_ratio)
    err_count = 0.0
    for i in range(num_test):
        classify_result = knn_classify(normalize_dating_data_matrix[i, :],
                                    normalize_dating_data_matrix[num_test: m,:],
                                    dating_labels[num_test:m], 7)
        print 'Classifier = %d, real answer = %d' % (classify_result, dating_labels[i]),
        if classify_result != dating_labels[i]:
            err_count += 1
            print '[X]'
        else:
            print '[O]'
    print "Total error rate = %.2f%%" % (err_count / float(num_test) * 100.0)

def classify_person():
    result_list = ['not at all', 'a little', 'very like']
    play_ratio = float(raw_input('Enter play ratio >> '))
    flier_mile = float(raw_input('Enter flier miles >> '))
    ice_cream = float(raw_input('Enter liters of ice cream >> '))

    dating_data_matrix, dating_labels = file2matrix('datingTestSet2.txt')
    normalize_dating_data_matrix, ranges, min_val = normalize(dating_data_matrix)
    input = array([flier_mile, play_ratio, ice_cream])
    classify_result = knn_classify((input - min_val) / ranges, normalize_dating_data_matrix,
                                   dating_labels, 3)
    print 'You will like this person = %s' % result_list[classify_result - 1]

if __name__ == '__main__':
    knn_classifier_test()
    classify_person()