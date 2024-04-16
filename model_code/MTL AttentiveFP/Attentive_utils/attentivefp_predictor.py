import torch
import numpy as np
import torch.nn as nn
from dgllife.model.gnn import AttentiveFPGNN
from .AttentiveFPReadout import AttentiveFPReadout

###设置设备
if torch.cuda.is_available():
    print("Use GPU")
    device = 'cuda'
else:
    print("Use CPU")
    device = 'cpu'

###定义模型框架
class AttentiveFPPredictor(nn.Module):
    def __init__(self,node_feat_size,edge_feat_size,num_layers=2,num_timesteps=2,graph_feat_size=200,
                 predictor_hidden_feats=128,n_tasks=5,dropout=0.):
        super(AttentiveFPPredictor,self).__init__()   ###调用父类构造函数

        ###提取节点的特征的类
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,graph_feat_size=graph_feat_size,dropout=dropout)
        ###聚合节点特征生成图特征的类
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,num_timesteps=num_timesteps,dropout=dropout)

        ###对图级别的特征进行预测的类
        self.predict = nn.Sequential(nn.Dropout(dropout),nn.Linear(graph_feat_size,predictor_hidden_feats),
                                     nn.ReLU(),nn.LayerNorm(predictor_hidden_feats),nn.Linear(predictor_hidden_feats,n_tasks))

    def forward(self,g,node_feats,edge_feats,get_node_weight=False,get_node_gradient=False):
        node_feats = self.gnn(g,node_feats,edge_feats) ###只有一层AttentiveFPGNN提取节点特征，输出更新后的节点特征，维度是(V,graph_feat_size)
        graph_feats = self.readout(g,node_feats,get_node_weight) ##有两层readout层提取图级别的信息，输出更新后的图特征，维度是(1,graph_feat_size)

        ###以下是可解释性分析方法
        ###注意力权重进行可解释性分析
        if get_node_weight:
            g_feats,node_weights = self.readout(g,node_feats,get_node_weight)  ###输出更新后的图特征和原子权重列表

            return self.predict(g_feats),node_weights
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            return self.predict(g_feats)