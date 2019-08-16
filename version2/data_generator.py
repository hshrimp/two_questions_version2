#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: wushaohong
@time: 2019/8/16 上午11:24
"""
from version2.config import maxlen
import numpy as np
from version2.utils import seq_padding


class DataGenerator:
    def __init__(self, data, tokenizer, batch_size=16 * 2):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

        self.tokenizer = tokenizer

    def __len__(self):
        return self.steps

    @staticmethod
    def generator_y(length1, length2):
        li = [0] * length1
        li.append(1)
        li.extend([0] * (length2 - 1))
        return li[:maxlen]

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                first = self.data[i]
                # 将单句问句a加入数据,其标签位置为len(a)
                x1, x2 = self.tokenizer.encode(first=first)
                y = self.generator_y(len(first), 0)
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                # 待拼接问句
                if len(X1) % 10 == 0:
                    second = self.data[-i]
                    # 将拼接句子（a+b）加入数据，其标签位置为len(a)
                    text = (first + second)[:maxlen]
                    x1, x2 = self.tokenizer.encode(first=text)
                    y = self.generator_y(len(first), len(second))
                    X1.append(x1)
                    X2.append(x2)
                    Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []
