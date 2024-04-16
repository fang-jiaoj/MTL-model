# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Readout for AttentiveFP
# pylint: disable= no-member, arguments-differ, invalid-name

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AttentiveFPReadout']

# pylint: disable=W0221
class GlobalPool(nn.Module):
    """该步骤是一个池化层，用于聚集原子的特征得到一个图级别的特征"""
    """One-step readout in AttentiveFP

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, feat_size, dropout):
        """feat_size: 输入节点特征、图特征和输出图表示的维度大小。
        dropout: 执行 dropout 的概率。"""
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * feat_size, 1),
            nn.LeakyReLU()
        )  ###通过线性层计算节点权重的对数概率
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, feat_size)
        )  ###对节点特征进行投影，使用线性层和 dropout
        self.gru = nn.GRUCell(feat_size, feat_size) ###使用 GRUCell 对图特征进行更新

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """Perform one-step readout

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():   ###创建了一个局部作用域，确保在该作用域内的所有操作不会影响外部的 g 对象，以防止修改全局图对象
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z') ##得到节点的权重
            g.ndata['hv'] = self.project_nodes(node_feats)  ###将节点特征投影到新的表示 'hv' 中

            g_repr = dgl.sum_nodes(g, 'hv', 'a') ###按节点权重加权求和，得到图的表示 g_repr
            context = F.elu(g_repr) ##将图表示 g_repr 输入到激活函数 ELU 中，得到上下文特征 context

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a'] ###返回更新后的图特征以及节点的权重
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class computes graph representations out of node features.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            ###self.readouts中包含多个 GlobalPool 模块。每个 GlobalPool 模块用于执行一步的图池化操作
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.

        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with g.local_scope(): ###创建一个局部作用域，确保在该作用域内的所有操作不会影响外部的 g 对象
            g.ndata['hv'] = node_feats ##将输入的节点特征 node_feats 存储在图的节点数据 'hv' 中
            g_feats = dgl.sum_nodes(g, 'hv') ##对节点特征进行求和，得到初始的图表示 g_feats，维度是图级别特征的大小

        if get_node_weight:
            node_weights = [] ##用于存储每个时间步的节点权重

        ##在每个时间步，将图表示 g_feats 和节点特征 node_feats 传递给 GlobalPool 模块
        for readout in self.readouts:
            if get_node_weight:
                ##返回更新后的图表示 g_feats 和节点权重，并将节点权重添加到 node_weights 中
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            ###最终，返回更新后的图表示 g_feats 和节点权重列表 node_weights
            return g_feats, node_weights
        else:
            return g_feats

