#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 09/01/2020 21:14 
@Author: XinZhi Yao 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def read_file(loss_acc_file):
    loss_list = []
    acc_list = []

    with open(loss_acc_file) as f:
        for line in f:
            l = line.strip().split('\t')
            loss_list.append(float(l[0]))
            acc_list.append(float(l[1])/100)
    return loss_list, acc_list

def draw_acc_curve(train_acc_list: list, valid_acc_list: list):
    if len(train_acc_list) != len(valid_acc_list):
        raise ValueError

    epoch = [i for i in range(len(train_acc_list))]
    plt.title('ACC curve')
    plt.plot(epoch, train_acc_list, color='green', label='train_acc')
    plt.plot(epoch, valid_acc_list, color='red', label='valid_acc')
    plt.legend()

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()


def draw_loss_curve(train_loss_list: list, valid_loss_list: list):
    if len(train_loss_list) != len(valid_loss_list):
        raise ValueError

    epoch = [i for i in range(len(train_loss_list))]
    plt.title('loss curve')
    plt.plot(epoch, train_loss_list, color='green', label='train_loss')
    plt.plot(epoch, valid_loss_list, color='red', label='valid_loss')
    plt.legend()

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()



if __name__ == '__main__':
    ques_train_loss_acc_file = 'model/ques_train_loss_acc.txt'
    quse_valid_loss_acc_file = 'model/ques_valid_loss_acc.txt'

    ag_train_loss_acc_file = 'model/ag_train_loss_acc.txt'
    ag_valid_loss_acc_file = 'model/ag_valid_loss_acc.txt'

    ques_train_loss_list, ques_train_acc_list = read_file(ques_train_loss_acc_file)
    ques_valid_loss_list, ques_valid_acc_list = read_file(quse_valid_loss_acc_file)

    ag_train_loss_list, ag_train_acc_list = read_file(ag_train_loss_acc_file)
    ag_valid_loss_list, ag_valid_acc_list = read_file(ag_valid_loss_acc_file)

    # draw_acc_curve(ques_train_acc_list, ques_valid_acc_list)
    # draw_acc_curve(ag_train_acc_list, ag_valid_acc_list)

    # draw_loss_curve(ques_train_loss_list, ques_valid_loss_list)
    draw_loss_curve(ag_train_loss_list, ag_valid_loss_list)