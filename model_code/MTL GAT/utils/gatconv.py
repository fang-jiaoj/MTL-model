"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import torch
# pylint: enable=W0235

class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        """in_feats: 输入特征的数量。
        out_feats: 输出特征的数量。
        num_heads: 多头注意力机制中头的数量。
        feat_drop: 输入特征的丢弃率（dropout）。
        attn_drop: 注意力权重的丢弃率。
        negative_slope: Leaky ReLU 的负斜率。
        residual: 是否使用残差连接。
        activation: 可选的激活函数。
        allow_zero_in_degree: 是否允许输入图中的节点出度为零。
        bias: 是否使用偏置项。"""
        super(GATConv, self).__init__() ###使用父类的构造函数
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        ###将输入特征的数量扩展为源节点特征和目标节点特征的元组
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        ###控制是否允许输入图中的节点出度为零
        ###如果输入特征 in_feats 是一个元组（tuple），则分别创建源节点和目标节点的线性层
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        ###注意力权重的可学习参数
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop) ###输入特征的 dropout 层
        self.attn_drop = nn.Dropout(attn_drop) ###注意力权重的 dropout 层
        self.leaky_relu = nn.LeakyReLU(negative_slope) ###Leaky ReLU 激活函数
        ###如果使用偏置项 bias，则创建一个可学习的参数 self.bias
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        ###这段代码是 GATConv 类的 reset_parameters 方法，该方法用于重新初始化模型参数

        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        """用于设置是否允许输入图中存在零入度节点（节点的入度为零）"""
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        ###graph: 输入的图数据,feat: 输入节点的特征,get_attention: 一个布尔值，控制是否返回注意力权重。
        ### 使用 local_scope() 创建了一个上下文管理器，该上下文中的任何操作都不会影响到原始图 graph
        with graph.local_scope():
            if not self._allow_zero_in_degree: ###如果不允许存在零入度节点，进入下一步的检查
                if (graph.in_degrees() == 0).any(): ###使用 in_degrees() 函数计算了每个节点的入度，并检查是否存在零入度节点
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple): ###检查输入特征是否是元组类型。元组通常用于表示源节点特征和目标节点特征
                h_src = self.feat_drop(feat[0]) ###对源节点特征应用特征丢弃操作
                h_dst = self.feat_drop(feat[1]) ###对目标节点特征应用特征丢弃操作
                ###如果模型没有属性 fc_src，则表示该模型没有专门为源节点和目标节点定义不同的线性变换，而是共享一个线性变换
                if not hasattr(self, 'fc_src'):
                    ###对源节点特征应用线性变换，并调整形状以匹配 num_heads 和 out_feats，形状是(N,n_heads,out_feats)
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else: ###如果模型有属性 fc_src，则该模型专门为源节点和目标节点定义不同的线性变换
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else: ###如果输入特征不是元组类型，表示模型只有一个线性变换，且源节点和目标节点共享相同的特征。
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()] ###从源节点的特征中选择块图中目标节点的特征
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1) ##得到源节点的权重
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1) ##得到目标节点的权重
            graph.srcdata.update({'ft': feat_src, 'el': el}) ###更新源节点的数据字典，添加源节点特征和注意力权重
            graph.dstdata.update({'er': er}) ## 更新目标节点的数据字典，添加目标节点的注意力权重
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e')) ##将源节点的注意力权重和目标节点的注意力权重相加，并存储在图的边数据中
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) ##对边的注意力进行 softmax 操作，得到最终的注意力权重
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft')) ## 使用计算得到的注意力权重进行消息传递，其中使用了点乘和求和的操作
            rst = graph.dstdata['ft'] ###获取目标节点的特征作为最终结果
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            ###如果设置了 get_attention 为 True，则返回节点特征和图的边注意力权重；否则，只返回节点特征
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst