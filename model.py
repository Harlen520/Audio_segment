import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp
from torch import lstm


class AudioSegmentation(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, num_class=2, dropout=None, rnn_style="lstm"):
        super().__init__()
        self.rnn_style = rnn_style.lower()
        rnn_dropout = dropout if dropout is not None else 0.1
        if self.rnn_style == "gru":
            self.encoder = nn.GRU(input_size, hidden_size, num_layers, direction="bidirect")
        elif self.rnn_style == "lstm":
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, direction="bidirect")
        else:
            print(f"Please input right rnn style, like [gru, lstm]")

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(hidden_size * 2, num_class)

        # self.apply(self.init_weights)
        
    def forward(self, input_features):

        output_states, _ = self.encoder(input_features)
        
        # cls_drop = self.dropout(output_states)
        logits = self.classifier(output_states) # [b, 300, 2]
        
        return logits
    
    # def init_weights(self, layer):
    #     """ Initialization hook """
    #     if isinstance(layer, (nn.Linear)):
    #         # only support dygraph, use truncated_normal and make it inplace
    #         # and configurable later
    #         if isinstance(layer.weight, paddle.Tensor):
    #             layer.weight.set_value(
    #                 paddle.tensor.normal(mean=0.0, std=0.02,
    #                     shape=layer.weight.shape))
        