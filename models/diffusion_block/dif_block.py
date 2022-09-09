import torch.nn as nn

from .forecast import Forecast
from .dif_model import STLocalizedConv
from ..decouple.residual_decomp import ResidualDecomp


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, use_pre=None, dy_graph=None, sta_graph=None, **model_args):
        """Diffusion block

        Args:
            hidden_dim (int): hidden dimension.
            forecast_hidden_dim (int, optional): forecast branch hidden dimension. Defaults to 256.
            use_pre (bool, optional): if use predefined graph. Defaults to None.
            dy_graph (bool, optional): if use dynamic graph. Defaults to None.
            sta_graph (bool, optional): if use static graph (the adaptive graph). Defaults to None.
        """

        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        # diffusion model
        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, use_pre=use_pre, dy_graph=dy_graph, sta_graph=sta_graph, **model_args)

        # forecast
        self.forecast_branch = Forecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        # backcast
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        # esidual decomposition
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        """Diffusion block, containing the diffusion model, forecast branch, backcast branch, and the residual decomposition link.

        Args:
            history_data (torch.Tensor): history data with shape [batch_size, seq_len, num_nodes, hidden_dim]
            gated_history_data (torch.Tensor): gated history data with shape [batch_size, seq_len, num_nodes, hidden_dim]
            dynamic_graph (list): dynamic graphs.
            static_graph (list): static graphs (the adaptive graph).

        Returns:
            torch.Tensor: the output after the decoupling mechanism (backcast branch and the residual link), which should be fed to the inherent model. 
                          Shape: [batch_size, seq_len', num_nodes, hidden_dim]. Kindly note that after the st conv, the sequence will be shorter.
            torch.Tensor: the output of the forecast branch, which will be used to make final prediction.
                          Shape: [batch_size, seq_len'', num_nodes, forecast_hidden_dim]. seq_len'' = future_len / gap. 
                          In order to reduce the error accumulation in the AR forecasting strategy, we let each hidden state generate the prediction of gap points, instead of a single point.
        """

        # diffusion model
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)
        # forecast branch: use the localized st conv to predict future hidden states.
        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        # backcast branch: use FC layer to do backcast
        backcast_seq = self.backcast_branch(hidden_states_dif)
        # residual decomposition: remove the learned knowledge from input data
        backcast_seq = backcast_seq
        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden
