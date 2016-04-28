#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'enginebai'

from numpy import *
import scipy.spatial.distance as distance

v1 = mat([1, 2, 3])
v2 = mat([4, 5, 6])

euclidean_distance = sqrt((v1 - v2) * (v1 - v2).T)
print(euclidean_distance)

manhattan_distance = sum(abs(v1 - v2))
print(manhattan_distance)

chebyshev_distance = abs(v1 - v2).max()
print(chebyshev_distance)

print(v1.shape)
cos_val = vdot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))
print(cos_val)

v = mat([[1, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1, 1, 1]])
hamming_distance = shape(nonzero(v[0] - v[1])[0])[0]
print(hamming_distance)

print(distance.pdist(v, 'jaccard'))

vector_matrix = mat([[1, 2, 3], [4, 5, 6]])
v12 = vector_matrix[0] - vector_matrix[1]
print(sqrt(v12 * v12.T))
var_mat = std(vector_matrix.T, axis=0)
normal_matrix = (vector_matrix - mean(vector_matrix)) / var_mat.T
normal_v12 = normal_matrix[0] - normal_matrix[1]
print(sqrt(normal_v12 * normal_v12.T)) 