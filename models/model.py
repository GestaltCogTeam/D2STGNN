import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SpatialBlock import SpaBlock
from models.TemporalBlock import TemBlock
from models.DynamicGraphConv.DyGraphCons import DynamicGraphConstructor
from models.Decouple.spatial_gate import SpatialGate

class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, first=False, **model_args):
        super().__init__()
        self.spatial_gate   = SpatialGate(model_args['node_hidden'], model_args['time_emb_dim'], 64, model_args['seq_length'])     # TODO: 测试Time Gate的hidden
        self.spa_layer      = SpaBlock(hidden_dim, fk_dim=fk_dim, **model_args)
        self.tem_layer      = TemBlock(hidden_dim, fk_dim=fk_dim, first=first, **model_args)

    def forward(self, X, dynamic_graph, static_graph, E_u, E_d, T_D, D_W):
        X_spa  = self.spatial_gate(E_u, E_d, T_D, D_W, X)
        spa_backcast_seq_res, spa_forecast_hidden = self.spa_layer(X=X, X_spa=X_spa, dynamic_graph=dynamic_graph, static_graph=static_graph)   
        tem_backcast_seq_res, tem_forecast_hidden = self.tem_layer(spa_backcast_seq_res)         
        return tem_backcast_seq_res, spa_forecast_hidden, tem_forecast_hidden

class DecoupleST(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self._in_feat       = model_args['num_feat']
        self._hidden_dim    = model_args['num_hidden']
        self._node_dim      = model_args['node_hidden']
        self._forecast_dim  = 256
        self._output_hidden = 512
        self._output_dim    = model_args['seq_length']

        self._num_nodes     = model_args['num_nodes']
        self._k_s           = model_args['k_s']
        self._k_t           = model_args['k_t']
        self._num_layers    = 5

        model_args['use_pre']   = False
        model_args['dy_graph']  = True
        model_args['sta_graph'] = True

        self._model_args    = model_args

        # start embedding layer
        self.embedding      = nn.Linear(self._in_feat, self._hidden_dim)

        # time embedding
        self.T_i_D_emb  = nn.Parameter(torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, first=True, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        # dynamic and static hidden graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor  = DynamicGraphConstructor(**model_args)
        
        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._num_nodes, self._node_dim))
 
        # output layer
        self.out_fc_1   = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2   = nn.Linear(self._output_hidden, model_args['gap'])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['E_d']
        E_u = inputs['E_u']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph   = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph   = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, X):
        num_feat    = self._model_args['num_feat']
        # node embeddings
        node_emb_u  = self.node_emb_u  # [N, d]
        node_emb_d  = self.node_emb_d  # [N, d]
        # time slot embedding
        T_i_D = self.T_i_D_emb[(X[:, :, :, num_feat] * 288).type(torch.LongTensor)]    # [B, L, N, d]
        D_i_W = self.D_i_W_emb[(X[:, :, :, num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d]
        # traffic signals
        X = X[:, :, :, :num_feat]

        return X, node_emb_u, node_emb_d, T_i_D, D_i_W

    def forward(self, X):
        r"""

        Args:
            X (Tensor): Input data with shape: [B, L, N, C]
        Returns:

        """
        # ==================== Prepare Input Data ==================== #
        X, E_u, E_d, T_D, D_W   = self._prepare_inputs(X)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(E_u=E_u, E_d=E_d, X=X, T_D=T_D, D_W=D_W)

        # Start embedding layer
        X   = self.embedding(X)

        spa_forecast_hidden_list = []
        tem_forecast_hidden_list = []

        tem_backcast_seq_res = X
        for index, layer in enumerate(self.layers):
            tem_backcast_seq_res, spa_forecast_hidden, tem_forecast_hidden = layer(tem_backcast_seq_res, dynamic_graph, static_graph, E_u, E_d, T_D, D_W)
            spa_forecast_hidden_list.append(spa_forecast_hidden)
            tem_forecast_hidden_list.append(tem_forecast_hidden)

        # Output Layer
        spa_forecast_hidden = sum(spa_forecast_hidden_list)
        tem_forecast_hidden = sum(tem_forecast_hidden_list)
        forecast_hidden     = spa_forecast_hidden + tem_forecast_hidden
        
        # regression layer
        forecast    = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast    = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)

        return forecast
