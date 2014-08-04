__author__ = 'enginebai'

from numpy import *
import operator

def create_data_set():
    group = array([1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1])
    labels = ['A', 'B', 'C', 'D']
    return group, labels

