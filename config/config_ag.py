#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 09/01/2020 10:48 
@Author: XinZhi Yao 
"""

params = {

    # model parameters
    'embed_dim': 64,
    'hidden_size': 32,
    'bidirectional': True,
    'weight_decay': 0.001,
    'momentum': 0,
    'attention_size': 16,
    # 'sequence_length': 20,
    'max_length': 75,
    'output_size': 6,

    # data parameters
    'seed': 1314,
    'use_cuda': False,
    'start_end_symbol': True,
    'label': True,
    'model_save_path': 'model/bilstm_attn_model_ag.pt',
    'logging_file': 'model/log_ag.txt',
    'train_data_path': 'data/AG_corpus_data/AG.train.txt',
    'valid_data_path': 'data/AG_corpus_data/AG.valid.txt',
    'train_loss_acc_save_file': 'model/ag_train_loss_acc.txt',
    'valid_loss_acc_save_file': 'model/ag_valid_loss_acc.txt',
}
