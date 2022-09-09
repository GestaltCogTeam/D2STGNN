import math

import torch
import torch.nn as nn

from models.decouple.residual_decomp import ResidualDecomp
from models.inherent_block.inh_model import RNNLayer, TransformerLayer
from models.inherent_block.forecast import Forecast


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        """Inherent block

        Args:
            hidden_dim (int): hidden dimension
            num_heads (int, optional): number of heads of MSA. Defaults to 4.
            bias (bool, optional): if use bias. Defaults to True.
            forecast_hidden_dim (int, optional): forecast branch hidden dimension. Defaults to 256.
        """
        super().__init__()
        self.num_feat   = hidden_dim
        self.hidden_dim = hidden_dim

        # inherent model
        self.pos_encoder    = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer          = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer  = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)
        
        # forecast branch
        self.forecast_block = Forecast(hidden_dim, forecast_hidden_dim, **model_args)
        # backcast branch
        self.backcast_fc    = nn.Linear(hidden_dim, hidden_dim)
        # residual decomposition
        self.residual_decompose   = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, hidden_inherent_signal):
        """Inherent block, containing the inherent model, forecast branch, backcast branch, and the residual decomposition link.

        Args:
            hidden_inherent_signal (torch.Tensor): hidden inherent signal with shape [batch_size, seq_len, num_nodes, num_feat].

        Returns:
            torch.Tensor: the output after the decoupling mechanism (backcast branch and the residual link), which should be fed to the next decouple layer. 
                          Shape: [batch_size, seq_len, num_nodes, hidden_dim]. 
            torch.Tensor: the output of the forecast branch, which will be used to make final prediction.
                          Shape: [batch_size, seq_len'', num_nodes, forecast_hidden_dim]. seq_len'' = future_len / gap. 
                          In order to reduce the error accumulation in the AR forecasting strategy, we let each hidden state generate the prediction of gap points, instead of a single point.
        """

        [batch_size, seq_len, num_nodes, num_feat]  = hidden_inherent_signal.shape
        # inherent model
        ## rnn
        hidden_states_rnn   = self.rnn_layer(hidden_inherent_signal)
        ## pe
        hidden_states_rnn   = self.pos_encoder(hidden_states_rnn)                   
        ## MSA
        hidden_states_inh   = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        # forecast branch
        forecast_hidden = self.forecast_block(hidden_inherent_signal, hidden_states_rnn, hidden_states_inh, self.transformer_layer, self.rnn_layer, self.pos_encoder)

        # backcast branch
        hidden_states_inh   = hidden_states_inh.reshape(seq_len, batch_size, num_nodes, num_feat)
        hidden_states_inh   = hidden_states_inh.transpose(0, 1)
        backcast_seq    = self.backcast_fc(hidden_states_inh)                                   
        backcast_seq_res= self.residual_decompose(hidden_inherent_signal, backcast_seq)                    

        return backcast_seq_res, forecast_hidden
