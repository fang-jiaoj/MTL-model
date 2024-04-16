import torch.nn  as nn
import torch.nn.functional as F
from .gatconv import GATConv

class GATLayer(nn.Module):
    """定义一个多头GAT层用于提取特征"""
    def __init__(self,in_feats,out_feats,num_heads,feat_drop,attn_drop,alpha=0.2,residual=True,agg_mode='flatten',
                 activation=None,bias=True):
        super(GATLayer,self).__init__()

        self.gat_conv = GATConv(in_feats=in_feats,out_feats=out_feats,num_heads=num_heads,feat_drop=feat_drop,attn_drop=attn_drop,
                                allow_zero_in_degree=True,negative_slope=alpha,residual=residual,bias=bias) ###多头GAT底层模型框架
        assert agg_mode in ['flatten','mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.gat_conv.reset_parameters()  ###该方法用于重新初始化模型参数

    def forward(self,bg,feats):
        """输入图对象以及节点的特征矩阵，输出经过一个多头GAT的特征矩阵和边权重"""
        feats,node_weights = self.gat_conv(bg, feats,get_attention=True)
        if self.agg_mode == 'flatten':
            feats = feats.flatten(1) ###将多个特征矩阵按列拼接
        else:
            feats = feats.mean(1)  ###将多个特征矩阵进行求均值

        if self.activation is not None:
            feats = self.activation(feats)
        return feats,node_weights

class GAT(nn.Module):
    """定义2层多头GAT层用于提取特征"""
    def __init__(self,in_feats,hidden_feats=None,num_heads=None,feat_drops=None,attn_drops=None,alphas=None,residuals=None,agg_modes=None,
                 activations=None,biases=None):
        super(GAT,self).__init__()

        if hidden_feats is None:
            hidden_feats = [32,32] ###设置一个多头GAT层的输出维度32

        ###设置各种超参数
        n_layers = len(hidden_feats) ###一共存在两层多头GAT层
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)] ##每层的头数是4
        if feat_drops is None:
            feat_drops = [0. for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0. for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)] ##每层leakyrelu的负斜率是0.2
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ['flatten' for _ in range(n_layers - 1)]  ##第一层的聚合不同头特征的方式是拼接，第二层是均值
            agg_modes.append('mean')
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)] ##第一层使用elu函数作为激活函数，第二层不使用激活函数
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]

        lengths = [len(hidden_feats), len(num_heads), len(feat_drops), len(attn_drops),
                   len(alphas), len(residuals), len(agg_modes), len(activations), len(biases)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, num_heads, ' \
                                       'feat_drops, attn_drops, alphas, residuals, ' \
                                       'agg_modes, activations, and biases to be the same, ' \
                                       'got {}'.format(lengths)
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GATLayer(in_feats, hidden_feats[i], num_heads[i],
                                            feat_drops[i], attn_drops[i], alphas[i],
                                            residuals[i], agg_modes[i], activations[i],
                                            biases[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()  ###通过迭代模型中的每个 GATLayer，并调用其 reset_parameters 方法来实现

    def forward(self,g,feats):
        for gnn in self.gnn_layers:
            feats,node_weights = gnn(g,feats)
        return feats,node_weights ###返回最后一层的输出特征以及边权重



