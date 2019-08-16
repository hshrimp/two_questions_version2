#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/16 下午2:45
"""

from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from version2.tokenizer_v2 import tokenizer
from version2.utils import seq_padding, get_data
import pandas as pd
from version2.config import maxlen, custom_dict


def generator_y(length1, length2):
    li = [0] * length1
    li.append(1)
    li.extend([0] * (length2 - 1))
    return li[:maxlen]


def generator_test_data(data):
    idxs = list(range(len(data)))
    np.random.shuffle(idxs)
    X1, X2, Y, texts = [], [], [], []
    for i in idxs:
        first = data[i]
        x1, x2 = tokenizer.encode(first=first)
        X1.append(x1)
        X2.append(x2)
        Y.append(len(first))
        texts.append(first)

        second = data[-i]
        text = (first + second)[:maxlen]
        x1, x2 = tokenizer.encode(first=text)
        X1.append(x1)
        X2.append(x2)
        Y.append(len(first))
        texts.append(text)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    return X1, X2, Y, texts


def write_to_csv(texts, Y, pred):
    table = pd.DataFrame()
    table['texts'] = texts
    table['text1'] = [t[:point] for t, point in zip(texts, pred)]
    table['text2'] = [t[point:] for t, point in zip(texts, pred)]
    table['Y'] = Y
    table["pred"] = pred

    table.to_csv('result_1vs10.csv', index=None)


if __name__ == '__main__':
    data_path = '/home/wushaohong/Downloads/lcqmc中文问句相似对/test.tsv'
    valid_data = get_data(data_path)
    X1, X2, Y, texts = generator_test_data(valid_data)
    model = load_model('model_v2_1.h5', custom_objects=custom_dict)
    pred = model.predict([X1, X2], batch_size=64)
    pred = np.argmax(pred, axis=1)
    print(accuracy_score(Y, pred))
    X1 = list(X1)
    pred = list(pred)
    write_to_csv(texts, Y, pred)
