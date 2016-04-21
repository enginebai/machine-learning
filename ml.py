#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'enginebai'

from numpy import *

vector1 = mat([1, 2, 3])
vector2 = mat([4, 5, 6])

euclidean_distance = sqrt((vector1 - vector2) * (vector1 - vector2).T)
print(euclidean_distance)

