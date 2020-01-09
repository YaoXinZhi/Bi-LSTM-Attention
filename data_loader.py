#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 07/01/2020 16:29 
@Author: XinZhi Yao 
"""

import os
import torch
import numpy as np

import random
from collections import defaultdict

# Done(20200108): Add start and end symbols
# Done: label2index label_class, turn lb_list to lb_index_list.
# Done: lower in pre_file()

def pre_file(input, output, lower=True):
    wf  = open(output, 'w')
    with open(input) as f:
        for line in f:
            l = line.strip().split(' ')
            label = l[0].split(':')[0]
            sent = ' '.join(l[1:])
            if lower:
                sent = sent.lower()
            wf.write('{0}\t{1}\n'.format(label, sent))
    wf.close()
    print('pre-process {0} done.'.format(os.path.basename(input)))

class VocabEntry(object):
    """docstring for Vocab"""
    def __init__(self, word2id=None, start_end_symbol=False):
        super(VocabEntry, self).__init__()

        if word2id:
            self.word2id = word2id
            self.unk_id = word2id['<unk>']
        else:
            self.word2id = dict()
            if start_end_symbol:
                self.word2id['<pad>'] = 0
                self.word2id['<s>'] = 1
                self.word2id['</s>'] = 2
                self.word2id['<unk>'] = 3
                self.unk_id = self.word2id['<unk>']
            else:
                self.word2id['<pad>'] = 0
                self.word2id['<unk'] = 1
                self.unk_id = self.word2id['<unk>']

        self.id2word_ = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word_[wid] = word
            return wid

        else:
            return self[word]

    def id2word(self, wid):
        return self.id2word_[wid]

    def decode_sentence(self, sentence):
        decoded_sentence = []
        for wid_t in sentence:
            wid = wid_t.item()
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence

    @staticmethod
    def from_corpus(fname):
        vocab = VocabEntry()
        with open(fname) as fin:
            for line in fin:
                _ = [vocab.add(word) for word in line.split()]

        return vocab

class LabelEntry(object):
    def __init__(self, lb2index=None):
        super(LabelEntry, self).__init__()

        if lb2index:
            self.lb2index = lb2index
        else:
            self.lb2index = dict()

        self.index2lb_ = {v: k for k, v in self.lb2index.items()}

    def __getitem__(self, lb):
        # if not self.lb2index.get(lb):
        #     print('lb: {0}'.format(lb))
        #     raise KeyboardInterrupt
        # else:
        #     return self.lb2index.get(lb, 'None')
        return self.lb2index.get(lb, 'None')

    def __contains__(self, lb):
        return lb in self.lb2index

    def __len__(self):
        return len(self.lb2index)

    def add(self, lb):
        if lb not in self:
            lb_index = self.lb2index[lb] = len(self)
            self.index2lb_ = word
            return lb_index
        else:
            return self[lb]

    def index2lb(self, lb_index):
        return self.index2lb_[lb]

class MonoTextData(object):
    def __init__(self, fname, max_length, label=False, vocab=None,
                 minfre=0, init_vocab_file=None, start_end_symbol=False, lb_entry=None):
        super(MonoTextData, self).__init__()
        self.label = label
        self.max_length = max_length
        self.select_max_length = max_length
        self.minfre = minfre
        self.start_end_symbol = start_end_symbol
        if self.start_end_symbol:
            self.select_max_length -= 2
        if init_vocab_file:
            print('inin vocab.')
            vocab = self._read_init_vocab(init_vocab_file, vocab)
        self.data, self.labels, self.vocab, self.lb_entry , self.dropped = self._read_corpus(fname, vocab, lb_entry)

    def __len__(self):
        return len(self.data)

    def _init_vocab(self):
        vocab = defaultdict(lambda: len(vocab))
        vocab['<pad>'] = 0
        if self.start_end_symbol:
            vocab['<s>'] = 1
            vocab['</s>'] = 2
            vocab['<unk>'] = 3
        else:
            vocab['<unk>'] = 1
        return vocab

    def _read_init_vocab(self, fname, vocab):
        print('init voacb from {0}'.format(fname))
        if not vocab:
            vocab = self._init_vocab()

        vocab_count_dic = defaultdict(int)
        with open(fname) as fin:
            for line in fin:
                if self.label:
                    split_line = line.split('\t')[1].split()
                else:
                    split_line = line.split()
                if len(split_line) < 1 or len(split_line) > self.select_max_length:
                    continue
                for word in split_line:
                    vocab_count_dic[word] += 1

        for word, value in vocab_count_dic.items():
            if value > self.minfre:
                index = vocab[word]
        if not isinstance(vocab, VocabEntry):
            vocab = VocabEntry(vocab, self.start_end_symbol)
        return vocab

    def _read_corpus(self, fname, vocab, lb_entry):

        data = []
        labels = [] if self.label else None
        dropped = 0
        vocab_count_dic = defaultdict(int)

        if not vocab:
            vocab = self._init_vocab()
        if not lb_entry:
            lb_entry = defaultdict(lambda: len(lb_entry))

        if self.minfre:
            with open(fname) as fin:
                for line in fin:
                    if self.label:
                        lb = line.split('\t')[0]
                        print(lb)
                        lb_index = lb_entry[lb]

                        split_line = line.split('\t')[1].split()
                    else:
                        split_line = line.split()
                    if len(split_line) < 1 or len(split_line) > self.select_max_length:
                        continue
                    for word in split_line:
                        vocab_count_dic[word] += 1
            for word, count in vocab_count_dic.items():
                if count > self.minfre:
                    # print(word, count)
                    index = vocab[word]
            if not isinstance(vocab, VocabEntry):
                vocab = VocabEntry(vocab)
            if not isinstance(lb_entry, LabelEntry):
                lb_entry = LabelEntry[lb_entry]

        with open(fname) as fin:
            for line in fin:
                if self.label:
                    split_line = line.split('\t')
                    lb = split_line[0]

                    split_line = split_line[1].split()
                else:
                    split_line = line.split()
                if len(split_line) < 1 or len(split_line) > self.select_max_length:
                    dropped += 1
                    continue
                if self.label:
                    labels.append(lb_entry[lb])
                data.append([vocab[word] for word in split_line])

        if not isinstance(vocab, VocabEntry):
            vocab = VocabEntry(vocab)
        if not isinstance(lb_entry, LabelEntry):
            lb_entry = LabelEntry(lb_entry)
        return data, labels, vocab, lb_entry, dropped

    def padding_to_fixlen(self, data):
        sents_len = np.array([len(sent) for sent in data])
        padded_sents_list = []
        for sent in data:
            if self.start_end_symbol:
                sent = [self.vocab['<s>']] + sent + [self.vocab['</s>']]
            if len(sent) < self.max_length:
                for _ in range(self.max_length - len(sent)):
                    sent += [self.vocab.word2id['<pad>']]
                padded_sents_list.append(sent)
            else:
                padded_sents_list.append(sent)
        return padded_sents_list


    def batch_iter(self, data, batch_size, labels=None, num_epochs=1, shuffle=True):
        data = self.padding_to_fixlen(data)
        data = np.array(data)
        if self.label:
            labels = np.array(labels)

        zip_data = []

        data_size = len(data)
        num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                if self.label:
                    shuffled_label = labels[shuffle_indices]
            else:
                shuffled_data = data
                if self.label:
                    shuffled_label = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index: end_index], shuffled_label[start_index: end_index]


# if __name__ == '__main__':
#     train_raw_file = 'data/train'
#     train_out_file = 'data/question.train.txt'
#
#     test_raw_file = 'data/valid'
#     test_out_file = 'data/question.valid.txt'
#     #
#     # pre_file(train_raw_file, train_out_file)
#     # pre_file(test_raw_file, test_out_file)
#
#     max_length = 20
#     batch_size = 16
#     start_end_symbol = True
#     label = True
#     epochs = 5
#
#     train_dataset = MonoTextData(train_out_file, label=label, max_length=max_length, start_end_symbol=start_end_symbol)
#     vocab = train_dataset.vocab
#     vocab_size = vocab.__len__()
#     lb_entry = train_dataset.lb_entry
#     # if label:
#     #     print('data size: {0}, dropped: {1}, vocab_size: {2}, label num: {3}'.format(len(train_dataset.data), train_dataset.dropped, vocab_size ,train_dataset.lb_entry.__len__()))
#     # else:
#     #     print('data size: {0}, dropped: {1}, vocab_size: {2}'.format(len(train_dataset.data), train_dataset.dropped, vocab_size))
#
#     if label:
#         print('train data size: {0}, dropped: {1}, test data size: {2}, vocab_size: {3}, label num: {4}'. \
#                 format(len(train_dataset.data), train_dataset.dropped, len(test_dataset.data), vocab_size,
#                        train_dataset.lb_entry.__len__()))
#     else:
#         print('data size: {0}, dropped: {1}, test data size: {2}, vocab_size: {3}, label num: {4}'. \
#                 format(len(train_dataset.data), train_dataset.dropped, len(test_dataset), vocab_size,
#                        train_dataset.lb_entry.__len__()))
#
#     test_dataset = MonoTextData(test_out_file, label=True, max_length=max_length, vocab=vocab,
#                                 start_end_symbol=start_end_symbol, lb_entry=lb_entry)
#
#     train_data_loader = train_dataset.batch_iter(train_dataset.data, batch_size=batch_size, labels=train_dataset.labels,
#                                                  num_epochs=epochs, shuffle=True)
#
#     test_data_loader = test_dataset.batch_iter(test_dataset.data, batch_size=batch_size, labels=test_dataset.labels,
#                                                num_epochs=1, shuffle=False)
#
#     # for i in test_data_loader:
#     count = 0
#     iter_nlog = np.floor(train_dataset.data.__len__() / batch_size)
#     report_batch = 0
#     report_epoch = 1
#     for i in train_data_loader:
#         report_batch += 1
#         if report_batch % iter_nlog == 0:
#             print('epoch: {0}, batch: {1}'.\
#                   format(report_epoch, report_batch))
#             report_epoch += 1
#
#         batch_data, batch_label = i
#         batch_data_tensor = torch.Tensor(batch_data).long()
#         # print(batch_data_tensor)
#         print(batch_data_tensor.shape)
#         batch_label_tensor = torch.Tensor(batch_label).long()
#         # print(batch_label_tensor)
#         print(batch_label_tensor.shape)
