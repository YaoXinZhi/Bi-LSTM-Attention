#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 09/01/2020 15:08 
@Author: XinZhi Yao 
"""

import os
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt

def sort_dic_key(dic: dict):
    sorted_dic = OrderedDict()
    key_sort = sorted(dic.keys(), key=lambda x: x, reverse=False)
    for key in key_sort:
        sorted_dic[key] = dic[key]
    # print(sorted_dic)
    return sorted_dic

def read_file(file):
    len_count_dic = defaultdict(int)
    label_count_dic = defaultdict(int)

    with open(file) as f:
        for line in f:
            l = line.strip().split('\t')
            test_len = len(l[1].split())
            # print(test_len)
            len_count_dic[test_len] += 1
            # print(len_count_dic)
            label_count_dic[l[0]] +=1
    # print(len_count_dic)
    len_count_sort_dic = sort_dic_key(len_count_dic)
    # print(len_count_sort_dic)

    return len_count_sort_dic, label_count_dic

def unzip_dic(count_dic: dict):
    len_list = []
    count_list = []
    for len, count in count_dic.items():
        len_list.append(len)
        count_list.append(count)
    return len_list, count_list

def draw_len_bar(train_len_dic: dict, test_len_dic: dict):
    train_len_list, train_count_list = unzip_dic(train_len_dic)
    test_len_list, test_count_list = unzip_dic(test_len_dic)


    plt.bar(train_len_list, train_count_list,label='train_data')
    plt.bar(test_len_list, test_count_list, label='valid_data')

    plt.legend()
    plt.xlabel('length')
    plt.ylabel('count')
    plt.title('sentence length statistics')

    plt.show()

def draw_label_dic(train_label_dic: dict, test_label_dic: dict):
    label_set = set()
    for key in train_label_dic.keys():
        label_set.add(key)
    label_list = list(label_set)

    train_count_list = [train_label_dic[key] for key in label_list]
    test_count_list = [test_label_dic[key] for key in label_list]


    plt.bar(label_list, train_count_list, label='train_data')
    plt.bar(label_list, test_count_list, label='valid_data')

    plt.legend()
    plt.xlabel('label')
    plt.ylabel('count')
    plt.title('Label count statistics.')

    plt.show()


def data_statistics(train_file, test_file):
    train_len_sort_dic, train_label_dic = read_file(train_file)
    test_len_sort_dic, test_label_dic = read_file(test_file)

    # draw_len_bar(train_len_sort_dic, test_len_sort_dic)

    draw_label_dic(train_label_dic, test_label_dic)




if __name__ == '__main__':

    ag_path = 'AG_corpus_data'
    ag_train_file = os.path.join(ag_path, 'AG.train.txt')
    ag_test_file = os.path.join(ag_path, 'AG.valid.txt')

    ques_path = 'question_clas'
    ques_train_file = os.path.join(ques_path, 'question.train.txt')
    ques_test_file = os.path.join(ques_path, 'question.valid.txt')

    # data_statistics(ag_train_file, ag_test_file)

    data_statistics(ques_train_file, ques_test_file)

