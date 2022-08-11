# basic
import os
from tqdm import tqdm

# np/pd
import numpy as np
import pandas as pd

# torch
import torch
import torchtext
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# transformer
from datasets import load_dataset

# CRF
from torchcrf import CRF




class BiLSTM_CRF(nn.Module):

    def __init__(self, config=None):
        super(BiLSTM_CRF, self).__init__()
        self.config = config

        # BiLSTM-model 给 emission 层定义参数
        self.embedding_dim = self.config.get('embedding_dim', 200)
        self.hidden_dim = self.config.get('hidden_dim', 200)
        self.vocab_size = self.config.get('vocab_size', 23623)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.target_size = self.config.get('num_tags', 9)
        self.num_layers = 3

        # lstm
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim//2,
                            num_layers=self.num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.target_size)
        self.hidden_init = self.init_hidden()

        # CRF-model
        self.crf = CRF(self.config.get('num_tags', 9), batch_first=True)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        cell = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        return hidden, cell

    def forward(self, sent, sent_len):
        """

        :param sent: 输入的已转换为token_id的句子，(batch_len * sent_len * token_emb_len)
        :param sent_len: tensor(list(int))
        :return:
        """
        embeds = self.word_embeds(sent)

        embed_packed = pack_padded_sequence(input=sent, lengths=sent_len,
                                            batch_first=True,
                                            enforce_sorted=False)
        lstm_out, (hidden, cell) = lstm(embed_packed, self.hidden_init)
        lstm_out, lens = pad_packed_sequence(lstm_out, batch_first=True)
        tag_score = self.hidden2tag(lstm_out)
        return tag_score


