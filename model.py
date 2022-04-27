import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp
from torch import lstm


class AudioSegmentation(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=None, rnn_style="lstm"):
        super().__init__()
        self.rnn_style = rnn_style.lower()
        if self.rnn_style == "gru":
            self.encoder = nn.GRU(input_size, hidden_size, num_layers, direction="bidirect")
        elif self.rnn_style == "lstm":
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, direction="bidirect")
        else:
            print(f"Please input right rnn style, like [gru, lstm]")

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(hidden_size * 2, 2)
        
    def forward(self, input_features):

        output_states, _ = self.encoder(input_features)
        cls_state = output_states[:, -1]
        cls_drop = self.dropout(cls_state)
        logits = self.classifier(cls_drop)
        
        return logits