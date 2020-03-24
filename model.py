import numpy as np # to handle matrix and data operation
import pandas as pd # to read csv and handle dataframe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


from sklearn.model_selection import train_test_split

from torchnlp.word_to_vector import FastText




vectors = FastText()

LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS


class ToxicClassifierModel(nn.Module):
    def __init__(self):
        super(ToxicClassifierModel, self).__init__()
        self.BiGRU = nn.GRU(300, hidden_size = LSTM_UNITS, bidirectional=True, num_layers=1)
        self.BiRNN = nn.RNN(input_size = 2 * LSTM_UNITS, hidden_size = LSTM_UNITS, bidirectional=True)
        self.hidden1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.hidden2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.hidden3 = nn.Linear(DENSE_HIDDEN_UNITS, 6)
    
    def forward(self, X):

        X = X.permute(0, 2, 1)
        
        X = F.dropout2d(X, 0.2, training=self.training)
       
        X = X.permute(0, 2, 1)
        
        X = self.BiGRU(X)

        X = self.BiRNN(X[0])

        X = X[0]

        X = torch.cat((torch.max(X, 1).values, torch.mean(X, 1)), 1)

        X = X.add(F.relu(self.hidden1(X)))
    
        X = X.add(F.relu(self.hidden2(X)))
        
        X = torch.sigmoid(self.hidden3(X))
        
 
        
        return X