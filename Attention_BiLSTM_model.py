#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 08/01/2020 15:32 
@Author: XinZhi Yao 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class bilstm_attn(nn.Module):
    def __init__(self, args):
        super(bilstm_attn, self).__init__()

        self.batch_size = args.batch_size
        self.output_size = args.output_size
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim
        self.bidirectional = args.bidirectional
        self.dropout = args.dropout
        self.use_cuda = args.use_cuda
        self.sequence_length = args.max_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=args.vocab['<pad>'])
        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(
            self.embed_dim,
            self.hidden_size,
            self.layer_size,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        if self.bidirectional:
            self.layer_size = self.layer_size * 2

        self.attention_size = args.attention_size
        if self.use_cuda:
            self.w_omega = torch.zeros(self.hidden_size * self.layer_size, self.attention_size).cuda()
            self.u_omega = torch.zeros(self.attention_size).cuda()
        else:
            self.w_omega = torch.zeros(self.hidden_size * self.layer_size, self.attention_size)
            self.u_omega = torch.zeros(self.attention_size)
        self.w_omega.requires_grad = True
        self.u_omega.requires_grad = True

        self.label = nn.Linear(self.hidden_size * self.layer_size, self.output_size)


    def attention_net(self, lstm_output):
        # lstm_output sequence_length, batch_size, hidden_size*layer_size
        # output_reshape [sequence_length * batch_size, hidden_size*layer_size]
        output_reshape = lstm_output.reshape(-1, self.hidden_size*self.layer_size)
        # attn_tanh [sequence_length * batch_size, attention_size]
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # attn_hidden_layer [sequence_legth*batch_size, 1] self.u_omega.reshape(-1, 1)
        attn_hidden_layer = torch.mm(attn_tanh, self.u_omega.reshape(-1, 1))
        # exps [batch_size, sequence_length]
        exps = torch.exp(attn_hidden_layer).reshape(-1, self.sequence_length)
        # alphas [batch_size, squence_length]
        alphas = exps / torch.sum(exps, 1).reshape(-1, 1)
        alphas_reshape = alphas.reshape(-1, self.sequence_length, 1)
        # state batch_size, squence_length, hidden_size*layer_size
        state = lstm_output.permute(1, 0, 2)
        # attn_output [batch_size, hidden_size*layer_size]
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, input_sentences, batch_size):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        if self.use_cuda:
            h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).cuda()
            c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).cuda()
        else:
            h_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size)
            c_0 = torch.zeros(self.layer_size, batch_size, self.hidden_size)


        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)

        logits = self.label(attn_output)
        return logits

