#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 07/01/2020 16:03 
@Author: XinZhi Yao 
"""

import os
import functools

def logging(s, log_path, log_=True):
    # write log file.
    print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    # logging = get_logger(log_path='log.txt')
    return functools.partial(logging, log_path=log_path, **kwargs)


def save_loss_acc_file(loss_list, acc_list, save_file):

    with open(save_file, 'w') as wf:
        for i in range(len(loss_list)):
            wf.write('{0}\t{1}\n'.format(loss_list[i], acc_list[i]))
    print('{0} save done.'.format(os.path.basename(save_file)))





