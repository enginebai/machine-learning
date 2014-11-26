#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

def create_trainset():
    """
    產生訓練集合。
    :return:
    """
    trainset_tf = dict()
    trainset_tf[u'陽明山'] = (15, 25, 0, 5, 8, 3)
    trainset_tf[u'中正紀念堂'] = (35, 40, 1, 3, 3, 2)
    trainset_tf[u'北港肉羹'] = (5, 0, 35, 50, 0, 0)
    trainset_tf[u'蚵仔煎'] = (1, 5, 32, 15, 0, 0)
    trainset_tf[u'圓山大飯店'] = (10, 5, 7, 0, 2, 30)
    trainset_tf[u'W Hotel'] = (5, 5, 5, 15, 8, 32)

    trainset_class = dict()
    trainset_class[u'陽明山'] = 'V'
    trainset_class[u'中正紀念堂'] = 'V'
    trainset_class[u'北港肉羹'] = 'D'
    trainset_class[u'蚵仔煎'] = 'D'
    trainset_class[u'圓山大飯店'] = 'A'
    trainset_class[u'W Hotel'] = 'A'

    return trainset_tf, trainset_class

def cosine_similarity(v1, v2):
    """
    計算兩個向量的正弦相似度。距離越近，相似度數值會越高。
    :param v1:
    :param v2:
    :return:
    """
    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
    for i in range(0, len(v1)):
        sum_xx += math.pow(v1[i], 2)
        sum_xy += v1[i] * v2[i]
        sum_yy += math.pow(v2[i], 2)

    return sum_xy / math.sqrt(sum_xx * sum_yy)

def knn_classify(input_tf, trainset_tf, trainset_class, k):
    """
    執行kNN分類演算法
    :param input_tf: 輸入向量
    :param trainset_tf: 訓練集合向量
    :param trainset_class: 訓練集合分類
    :param k: 取k個最近鄰居
    :return:
    """
    tf_distance = dict()
    # 計算每個訓練集合特徵關鍵字字詞頻率向量和輸入向量的距離
    print '(1) 計算向量距離'
    for place in trainset_tf.keys():
        tf_distance[place] = cosine_similarity(trainset_tf.get(place), input_tf)
        print '\tTF(%s) = %f' % (place, tf_distance.get(place))

    # 把距離排序，取出k個最近距離的分類
    class_count = dict()
    print '(2) 取K個最近鄰居的分類, k = %d' % k
    for i, place in enumerate(sorted(tf_distance, key=tf_distance.get, reverse=True)):
        current_class = trainset_class.get(place)
        print '\tTF(%s) = %f, class = %s' % (place, tf_distance.get(place), current_class)
        class_count[current_class] = class_count.get(current_class, 0) + 1
        if (i + 1) >= k:
            break

    print '(3) K個最近鄰居分類出現頻率最高的分類當作最後分類'
    input_class = ''
    for i, c in enumerate(sorted(class_count, key=class_count.get, reverse=True)):
        if i == 0:
            input_class = c
        print '\t%s, %d' % (c, class_count.get(c))
    print '(4) 分類結果 = %s' % input_class

if __name__ == '__main__':
    input_tf = (10, 2, 50, 56, 8, 5)
    trainset_tf, trainset_class = create_trainset()
    knn_classify(input_tf, trainset_tf, trainset_class, k=3)