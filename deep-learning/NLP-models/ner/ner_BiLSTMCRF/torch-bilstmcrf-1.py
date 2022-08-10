# np/pd
import numpy as np
import pandas as pd

# torch
import torch
import torchtext
import torch.nn as nn

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
        self.vocab_size = self.config.get('vocab_size')

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.target_size = self.config.get('num_tags', 9)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim//2,
                            num_layers=3, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.target_size)

        # CRF-model
        self.crf = CRF(self.config.get('num_tags', 9), batch_first=True)

    def forward(self, sentence):
        '''
        input:

        :return:
        '''
        embs = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out)
