__author__ = 'enginebai'

from numpy import *
import operator

def create_data_set():
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def knn_classify(input, data_set, labels, k):
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




