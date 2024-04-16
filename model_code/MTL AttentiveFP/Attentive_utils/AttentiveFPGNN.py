# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AttentiveFP
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl.function as fn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import edge_softmax

__all__ = ['AttentiveFPGNN']

# pylint: disable=W0221, C0103, E1101
class AttentiveGRU1(nn.Module):
    """使用拥有注意力机制的边特征和之前的节点特征来更新节点特征"""
    """Update node features with attention and GRU.

    This will be used for incorporating the information of edge features
    into node features for message passing.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        """node_feat_size：输入节点特征的大小。edge_feat_size：输入边特征（也称为连接）的大小。edge_hidden_size：边隐藏层的维度。
        dropout：执行 dropout 的概率。"""
        super(AttentiveGRU1, self).__init__()  ###继承父类的构造函数

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        #####定义一个GRU层，它接受隐藏层边作为输入，并输出节点特征的大小
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """初始化边特征变换的线性层和 GRU 单元的参数"""
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """g: DGLGraph，表示一个batch的大图;edge_logits: 形状为 (E, 1) 的浮点数张量，其中 E 代表边的数量;
        edge_feats：表示先前的边特征，node_feats：表示先前的节点特征"""
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var() ##在原图上创建一个局部变量，以确保不影响全局图结构
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        ###通过 edge_softmax 函数对边 logits 进行 softmax 操作，得到边的权重。然后，将这些权重乘以经过边特征转换的边特征
        g.update_all(essage_func=dgl.function.u_copy_e('e', 'm'), reduce_func=dgl.function.sum('m', 'c'))
        ###使用 update_all 函数将边特征传递到节点，采用 "copy_edge" 函数将边的信息e传递到目标节点m(中间节点)，
        # 采用 "sum" 函数将目标节点的信息进行汇总，得到每个节点的上下文信息(得到聚合后的边数据 'c')
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats)) ###将边特征和节点特征结合起来更新节点特征

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    This will be used in GNN layers for updating node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        """node_feat_size: 输入节点特征的维度大小。
        edge_hidden_size: 中间边（键合）表示的维度大小。
        dropout: 执行 dropout 的概率。"""
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """重新初始化模型参数，包括节点特征变换的线性层和 GRU 单元"""
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var() ###.local_var() 是为了在原图上创建一个局部变量，以确保不影响全局图结构
        g.edata['a'] = edge_softmax(g, edge_logits) ###对边 logits 进行 softmax 操作，得到边的权重
        g.ndata['hv'] = self.project_node(node_feats)  ####得到变换后的节点特征

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        ###使用 update_all 函数将节点特征投影到边上，
        # 采用 "src_mul_edge" 函数将源节点的特征与边的权重相乘，并传递到目标节点，采用 "sum" 函数将目标节点的信息进行汇总，得到每个节点的上下文信息。
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats)) ###将上下文表示和先前的节点特征输入到 GRU 单元中，通过激活函数获得更新后的节点特征。

class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.

    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        """node_feat_size: 输入节点特征的维度大小。
        edge_feat_size: 输入边（键合）特征的维度大小。
        graph_feat_size: 学习到的图表示（分子指纹）的维度大小。
        dropout: 执行 dropout 的概率。"""
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def reset_parameters(self):
        """重新初始化模型参数，包括节点特征变换、边特征变换、边权重变换和 AttentiveGRU1 的参数"""
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        """用于对边进行更新，将源节点的特征和边特征连接起来"""
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """用于再次对边进行更新，将目标节点的新特征和之前更新的边特征连接起来"""
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        ###使用局部变量 g，将节点特征、新节点特征(维度大小是图级别的特征)、边特征分别保存在 'hv'、'hv_new' 和 'he' 中
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1) ##得到更新边特征 'he1'
        g.edata['he1'] = self.project_edge1(g.edata['he1'])  ###得到更新后的边特征，维度是图级别特征的大小
        g.apply_edges(self.apply_edges2)  ###得到更新后的边特征，维度是2*图特征的大小
        logits = self.project_edge2(g.edata['he2'])  ###获得边的权重，维度是(边数,1)

        ###将边权重信息和节点的上下文信息输入，更新节点特征
        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    """GNNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        """node_feat_size: 输入节点特征的维度大小。
        graph_feat_size: 要计算的图表示的维度大小。
        dropout: 执行 dropout 的概率。"""
        super(GNNLayer, self).__init__()

        ###将目标节点和源节点的特征连接后映射到权重的层
        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        ###将边特征信息和节点特征输入，更新节点特征
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)

    def reset_parameters(self):
        """初始化边特征变换和 AttentiveGRU2 的参数"""
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """用于对边进行更新，生成边特征，将目标节点和源节点的特征连接在一起"""
        """Edge feature generation.

        Generate edge features by concatenating the features of the destination
        and source nodes.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats  ##将节点特征保存在 'hv' 中
        g.apply_edges(self.apply_edges)  ##使用 apply_edges 方法更新边特征 'he'
        logits = self.project_edge(g.edata['he'])  ##得到边的权重

        return self.attentive_gru(g, logits, node_feats) ###模型更新节点特征

class AttentiveFPGNN(nn.Module):
    """`Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class performs message passing in AttentiveFP and returns the updated node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GNN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(AttentiveFPGNN, self).__init__()
        """node_feat_size: 输入节点特征的维度大小。
    edge_feat_size: 输入边特征的维度大小。
    num_layers: GNN 层的数量，默认为 2。
    graph_feat_size: 要计算的图表示的维度大小，默认为 200。
    dropout: 执行 dropout 的概率，默认为 0"""

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        ###使用 GetContext 层进行初始上下文的生成，该层通过边特征和节点特征更新节点表示
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        ###使用 init_context 层生成初始节点表示
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)  ###返回最终更新后的节点表示
        return node_feats
