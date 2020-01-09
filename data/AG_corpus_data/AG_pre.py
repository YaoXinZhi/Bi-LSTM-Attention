#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 09/01/2020 9:57 
@Author: XinZhi Yao 
"""

from collections import defaultdict, OrderedDict
from string import punctuation
import matplotlib.pyplot as plt

def unzip_dic(count_dic: dict):
    length_list = []
    count_list = []
    for len, count in count_dic.items():
        length_list.append(len)
        count_list.append(count)
    return length_list, count_list


def draw_len_dis(train_len_count_dic: dict, test_len_count_dic: dict):

    train_len_list, train_count_list = unzip_dic(train_len_count_dic)
    test_len_list, test_count_list = unzip_dic(test_len_count_dic)

    plt.bar(train_len_list, train_count_list, label='train_data')
    plt.bar(test_len_list, test_count_list, label='valid_data')

    plt.legend()
    plt.xlabel('length')
    plt.ylabel('count')
    plt.title('Sentence length statistics')

    plt.show()



def AG_pre(text_file, label_file, out_file):

    with open(text_file) as f:
        text_list = []
        length_dic = defaultdict(int)
        length_sort_dic = OrderedDict()
        for line in f:
            l = line.strip().lower()
            for punc in punctuation:
                text = l.replace(punc, ' ')
            length_dic[len(text.split())] += 1
            text_list.append(text)

    length_sort = sorted(length_dic.keys(), key=lambda x: x, reverse=False)
    for key in length_sort:
        length_sort_dic[key] = length_dic[key]

    data_size = len(text_list)

    with open(label_file) as f:
        label_list = []
        for line in f:
            l = line.strip()
            label_list.append(l)
    label_size = len(label_list)
    label_set = set(label_list)

    print('data_size: {0} | label_size: {1}'.format(data_size, label_size))
    print('length_len: {0}'.format(length_sort_dic))
    print('label: {0}'.format(label_set))
    print('-' * 90)

    if data_size != label_size:
        raise ValueError

    with open(out_file, 'w') as wf:
        for idx in range(data_size):
            wf.write('{0}\t{1}\n'.format(label_list[idx], text_list[idx]))
    print('save done.')

    return length_sort_dic




if __name__ == '__main__':
    train_text_file = 'train_texts.txt'
    train_label_file = 'train_labels.txt'
    train_out_file = 'AG.train.txt'

    test_label_file = 'test_labels.txt'
    test_text_file = 'test_texts.txt'
    test_out_file = 'AG.valid.txt'


    train_length_dic = AG_pre(train_text_file, train_label_file, train_out_file)
    test_length_dic =  AG_pre(test_text_file, test_label_file, test_out_file)
    print(train_length_dic)
    print(test_length_dic)
    draw_len_dis(train_length_dic, test_length_dic)