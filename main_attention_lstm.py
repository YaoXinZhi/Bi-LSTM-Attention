#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 07/01/2020 15:04
@Author: XinZhi Yao
"""

import os
import time
import importlib
import argparse
from tqdm import tqdm

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_logger, save_loss_acc_file
from data_loader import MonoTextData
import Attention_BiLSTM_model

logging = None


def init_config():
    parser = argparse.ArgumentParser(description='Bi-LSTM + Attention model for text classification.')
    parser.add_argument('--dataset', type=str, choices=['ques', 'ag'], required=True, help='dataset config file.')
    parser.add_argument('--lr', type=float, default=0.001, required=False, help='learning rate.')
    parser.add_argument('--epochs', type=int, default=1000, required=False, help='number of epoch.')
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='size of mini batch data.')
    parser.add_argument('--dropout', type=float, default=0.5, required=False, help='dropout rate.')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default='adam', required=False, help='optim.')
    parser.add_argument('--load_path', type=str, default=None, required=False, help='load model path.')
    parser.add_argument('--save_loss_acc',action='store_true', default=False, help='where save train loss and train acc.')
    args = parser.parse_args()

    config_file = 'config.config_{0}'.format(args.dataset)
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)
    args.vocab = None
    args.vocab_size = None

    args.use_cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    return args

def evaluate(model, valid_dataset, batch_size, use_cuda, compute_f=False):
    criterion = torch.nn.CrossEntropyLoss()
    corrects = eval_loss = 0
    report_size = valid_dataset.data.__len__()
    valid_data_loader = valid_dataset.batch_iter(valid_dataset.data, batch_size=batch_size,
                                               labels=valid_dataset.labels, num_epochs=1, shuffle=False)

    pred_label_list = []
    ture_label_list = []

    for mini_batch in valid_data_loader:
        batch_data, batch_label = mini_batch
        batch_data_tensor = torch.Tensor(batch_data).long()
        batch_label_tensor = torch.Tensor(batch_label).long()
        if use_cuda:
            batch_data_tensor = batch_data_tensor.cuda()
            batch_label_tensor = batch_label_tensor.cuda()
        batch_size, _ = batch_data_tensor.shape
        pred = model(batch_data_tensor, batch_size=batch_size)
        loss = criterion(pred, batch_label_tensor)

        eval_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(batch_label_tensor.size()).data == batch_label_tensor).sum()

        pred_label_list += torch.max(pred, 1)[1].view(batch_label_tensor.size()).cpu().numpy().tolist()
        ture_label_list += batch_label_tensor.cpu().numpy().tolist()

    if compute_f:
        precision_score = metrics.precision_score(ture_label_list, pred_label_list, average='macro')
        recall_score = metrics.recall_score(ture_label_list, pred_label_list, average='macro')
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
        logging('precision: {0:.4f} | recall: {1:.4f} | F_score: {2:.4f}'.\
                format(precision_score, recall_score, f1_score))
    return eval_loss / report_size, corrects, corrects*100.0/ report_size, report_size

def main(args):

    global logging
    logging = get_logger(args.logging_file)

    if args.use_cuda:
        logging('using cuda')
    logging(str(args))


    # load training data and valid data
    train_dataset = MonoTextData(args.train_data_path, label=args.label, max_length=args.max_length,
                                 start_end_symbol=args.start_end_symbol)
    args.vocab = train_dataset.vocab
    args.vocab_size = args.vocab.__len__()
    lb_entry = train_dataset.lb_entry
    valid_dataset = MonoTextData(args.valid_data_path, label=args.label, max_length=args.max_length, vocab=args.vocab,
                             start_end_symbol=args.start_end_symbol, lb_entry=lb_entry)

    train_data_loader = train_dataset.batch_iter(train_dataset.data, batch_size=args.batch_size,
                                                 labels=train_dataset.labels, num_epochs=args.epochs)
    if args.label:
        logging('train data size: {0}, dropped: {1}, valid data size: {2}, vocab_size: {3}, label num: {4}'.\
              format(len(train_dataset.data), train_dataset.dropped, len(valid_dataset.data) , args.vocab_size ,
                    train_dataset.lb_entry.__len__()))
    else:
        logging('data size: {0}, dropped: {1}, valid data size: {2}, vocab_size: {3}, label num: {4}'.\
              format(len(train_dataset.data), train_dataset.dropped, len(valid_dataset) , args.vocab_size,
                     train_dataset.lb_entry.__len__()))

    # init model
    bilstm_attn = Attention_BiLSTM_model.bilstm_attn(args)
    logging('init model done.')

    if not args.load_path is None:
        if args.use_cuda:
            bilstm_attn.load_state_dict(torch.load(args.load_path))
        else:
            bilstm_attn.load_state_dict(torch.load(args.load_path, map_location='cpu'))
        loss, corrects, acc, valid_dataset_size = evaluate(bilstm_attn, valid_dataset, args.batch_size, args.use_cuda)
        logging('loed model: loss {0:.4f} | accurcy {1}%({2}/{3})'. \
                format(loss, acc, corrects, valid_dataset_size))

    if args.use_cuda:
        bilstm_attn = bilstm_attn.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(bilstm_attn.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = optim.Adam(bilstm_attn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer not supported.')

    criterion = torch.nn.CrossEntropyLoss()

    # train, evaluate and save best model.
    best_acc = 0

    eval_niter = np.floor(train_dataset.data.__len__() / args.batch_size)
    log_niter = np.floor(eval_niter / 10.0)

    train_loss = 0
    train_corrects = 0
    report_epoch = 1
    report_size = 0

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    try:
        logging('-'*90)
        n_iter = 0
        logging_start_time = time.time()
        epoch_start_time = time.time()
        total_start_time = time.time()
        for mini_batch in train_data_loader:

            n_iter += 1
            batch_data, batch_label = mini_batch
            report_size += len(batch_data)
            batch_data_tensor = torch.Tensor(batch_data).long()
            batch_label_tensor = torch.Tensor(batch_label).long()
            if args.use_cuda:
                batch_data_tensor = batch_data_tensor.cuda()
                batch_label_tensor = batch_label_tensor.cuda()


            batch_size, _ = batch_data_tensor.shape
            target = bilstm_attn(batch_data_tensor, batch_size=batch_size)
            loss = criterion(target, batch_label_tensor)
            corrects = (torch.max(target, 1)[ 1 ].view(batch_label_tensor.size()).data == batch_label_tensor).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_corrects += corrects
            # todo: Visualization of attention weights.
            if n_iter % log_niter == 0:
                logging_end_time = time.time()
                report_loss = train_loss / report_size
                report_acc = train_corrects * 100 / report_size
                batch = n_iter - (report_epoch - 1) * eval_niter


                logging('epoch-batch {0}-{1} | cost_time {2:2.2f} | train_loss: {3:5.6f} | train_acc: {4}%'.\
                        format(report_epoch, int(batch), logging_end_time-logging_start_time, report_loss, report_acc))

                bilstm_attn.eval()
                eval_loss, eval_corrects, eval_acc, eval_valid_size = evaluate(bilstm_attn, valid_dataset, args.batch_size, args.use_cuda)
                bilstm_attn.train()
                train_loss_list.append(report_loss)
                train_acc_list.append(float(report_acc))
                valid_loss_list.append(eval_loss)
                valid_acc_list.append(float(eval_acc))

                train_loss = report_size = 0
                train_corrects = 0
                logging_start_time = logging_start_time

            if n_iter % eval_niter == 0:
                epoch_end_time = time.time()
                bilstm_attn.eval()
                # report_loss = train_loss / report_size
                eval_loss, corrects, acc, valid_dataset_size = evaluate(bilstm_attn, valid_dataset, args.batch_size, args.use_cuda)
                logging('-' * 10)
                logging('end_epoch {0:3d} | cost_time {1:2.2f} s | eval_loss {2:.4f} | accurcy {3}%({4}/{5})'.\
                      format(report_epoch, epoch_end_time-epoch_start_time, eval_loss, acc, corrects, valid_dataset_size))

                # train_loss = report_size = 0
                report_epoch += 1
                epoch_start_time = time.time()
                bilstm_attn.train()

                if best_acc < acc:
                    best_acc = acc
                    logging('update best acc: {0}%'.format(best_acc))
                    torch.save(bilstm_attn.state_dict(), args.model_save_path)
                logging('-'*10)
    except KeyboardInterrupt:
        logging('_'*90)
        logging('Exiting from training early.')
        bilstm_attn.eval()
        logging('load best model.')
        if args.use_cuda:
            bilstm_attn.load_state_dict(torch.load(args.model_save_path))
        else:
            bilstm_attn.load_state_dict(torch.load(args.model_save_path, map_location='cpu'))
        loss, corrects, acc, valid_dataset_size = evaluate(bilstm_attn, valid_dataset, args.batch_size, args.use_cuda, compute_f=True)
        logging('total_epoch {0:3d} | total_time {1:2.2f} s | loss {2:.4f} | accurcy {3}%({4}/{5})'. \
              format(report_epoch, time.time() - total_start_time, loss, acc, corrects, valid_dataset_size))
        logging('-'*90)
        if args.save_loss_acc:
            save_loss_acc_file(train_loss_list, train_acc_list, args.train_loss_acc_save_file)
            save_loss_acc_file(valid_loss_list, valid_acc_list, args.valid_loss_acc_save_file)


if __name__ == '__main__':
    args = init_config()
    main(args)

