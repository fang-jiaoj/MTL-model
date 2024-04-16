import torch
import torch.nn as nn
import numpy as np
from .gat_attention import GAT  ###对于包下的其他文件，使用相对路径导入
from .mlp_predictor_grad import MLPPredictor
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

###选定设备
if torch.cuda.is_available():
    print("Use GPU")
    device = 'cuda'
else:
    print("Use CPU")
    device = 'cpu'

class GATPredictor(nn.Module):
    """该函数使用GAT进行特征提取，然后使用MLP进行最终预测"""
    def __init__(self,in_feats,hidden_feats=None,num_heads=None,feat_drops=None,attn_drops=None,alphas=None,residuals=None,
                 agg_modes=None,activations=None,biases=None,classifier_hidden_feats=128,classifier_dropout=0.,n_tasks=5,
                 predictor_hidden_feats=128,predictor_dropout=0.):
        super(GATPredictor).__init__() ##使用父类的构造函数

        ###这些检查和调整确保了该类可以使用旧的参数名（classifier_hidden_feats和classifier_dropout）
        # 或新的参数名（predictor_hidden_feats和predictor_dropout）。
        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        ###定义GAT对象，进行特征提取（该类默认含有两个多头GAT层）
        self.gnn = GAT(in_feats=in_feats,hidden_feats=hidden_feats,num_heads=num_heads,feat_drops=feat_drops,attn_drops=attn_drops,
                       alphas=alphas,residuals=residuals,agg_modes=agg_modes,activations=activations,biases=biases)
        ###定义最后一层的输出维度
        if self.gnn.add_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.read_out = WeightedSumAndMax(gnn_out_feats) ##使用该函数将原子级别的特征聚合成图级别的特征，从而进行图级别的预测
        ###定义一个前馈神经网络用于最终的预测
        self.predict = MLPPredictor(2 * gnn_out_feats,predictor_hidden_feats,n_tasks,predictor_dropout)

    def forward(self,bg,feats,get_edge_weight=False,get_node_gradient=False):
        """该类通过get_edge_weight=False,get_node_gradient=False两个参数的取值可控制是否使用注意力或积分梯度解释"""
        node_feats,node_weights = self.gnn(bg,feats) ##经过两个多头GAT得到的特征和边权重
        graph_feats = self.read_out(bg,node_feats) ##得到图级别的特征

        ###使用注意力权重解释
        if get_edge_weight:
            Final_feature = self.predict(graph_feats)
            return Final_feature,node_weights ##返回预测结果（值）和边权重
        else:
            Final_feature = self.predict(graph_feats)
            return Final_feature