#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'enginebai'

import jieba
import jieba.posseg as pseg

jieba.set_dictionary('dict_zhTW.txt')
jieba.initialize()
sentence = input(">> ")
while sentence is not 'quit' and sentence:
    seg_list = jieba.cut(sentence)
    words = pseg.cut(sentence)
    print(" / ".join(seg_list))
    for w in words:
        print('{}[{}] '.format(w.word, w.flag), end=' ')
    print()
    sentence = input(">> ")