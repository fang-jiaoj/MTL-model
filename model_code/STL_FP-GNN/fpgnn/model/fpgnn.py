from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from fpgnn.data import GetPubChemFPs, create_graph, get_atom_features_dim
import csv

atts_out = []

class FPN(nn.Module):
    """定义一个指纹神经网络的类"""
    def __init__(self,args): #参数的解析后的命令行参数
        """该函数用于定义一些参数和层"""
        super(FPN, self).__init__()
        self.fp_2_dim=args.fp_2_dim  #FPN 中第二个层的维度大小
        self.dropout_fpn = args.dropout #FPN中的dropout比例
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size #模型隐藏层的维度
        self.args = args
        if hasattr(args,'fp_type'): #检查args中是否包含fp_type属性['morgan','mixed']
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'
        
        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024
        
        if hasattr(args,'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None
        
        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim) #定义一个全连接层，输入的维度是指纹的维度，输出的维度是fp_2_dim
        self.act_func = nn.ReLU() #创建一个激活函数ReLU，进行非线性变换，增加模型的表达能力
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim) #再次创建一个全连接层，输入的维度是fp_2_dim，输出的维度是hidden_size
        self.dropout = nn.Dropout(p=self.dropout_fpn) #创建一个dropout层，其中 p=self.dropout_fpn 表示 dropout 的概率
    
    def forward(self, smile):
        """定义一个前向传播函数"""
        fp_list=[]
        for i, one in enumerate(smile):
            fp=[]
            mol = Chem.MolFromSmiles(one)
            #根据self.fp_type的值，进而使用不同的方法计算指纹
            if self.fp_type == 'mixed': #计算混合指纹（融合三种指纹）
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol) #计算MACCS指纹
                #PhaErGFP指纹是一种结合了药效团信息的分子指纹，对于计算药物相似性和药物活性预测等任务具有一定的优势
                fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
                fp_pubcfp = GetPubChemFPs(mol) #计算PubChem指纹
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else: #计算Morgan指纹
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) #计算出1024位的Morgan指纹
                fp.extend(fp_morgan)
            fp_list.append(fp) #得到所有smiles指纹的列表

        #在处理某些特定的分子指纹情况下，需要对指纹进行一些修改
        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list) #维度：分子数*指纹的维度
            fp_list[:,self.fp_changebit-1] = np.ones(fp_list[:,self.fp_changebit-1].shape) #将索引为self.fp_changebit-1的列值设置为全1
            fp_list.tolist()
        
        fp_list = torch.Tensor(fp_list) #转变为Tensor，可支持GPU计算

        if self.cuda: #Cuda为True
            fp_list = fp_list.cuda() #张量.cuda()就可以在GPU上计算
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out #得到原特征矩阵经过FPN后的结果

class GATLayer(nn.Module):
    """定义一个单头的GAL层"""
    def __init__(self, in_features, out_features, dropout_gnn, alpha, inter_graph, concat=True):
        """定义一些层和参数"""
        #in_features:输入节点的特征维度，dropout_gnn：GAT中dropout的概率，alpha: LeakyReLU的负斜率
        #inter_graph:存储注意力权重的值,concat:使用拼接方式进行特征融合
        super(GATLayer, self).__init__()
        self.dropout_gnn= dropout_gnn
        self.in_features = in_features 
        self.out_features = out_features
        self.alpha = alpha 
        self.concat = concat 
        self.dropout = nn.Dropout(p=self.dropout_gnn) #定义一个Dropout层对象
        self.inter_graph = inter_graph

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        #self.W是一个可学习的参数矩阵，大小为 (in_features, out_features) 的零矩阵
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #对GAT层中的self.W参数进行Xavier初始化，以提供一个合理的初始权重，有助于更好地训练模型
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        #self.a也是一个可学习的参数矩阵，用于计算节点之间的注意力得分，大小为 (2*out_features, 1) 的零矩阵
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #使用 Xavier 初始化方法进行权重初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha) #定义一个LeakyReLU对象
        if self.inter_graph is not None:
            self.atts_out = [] #用于记录注意力权重值的列表
    
    def forward(self,mole_out,adj):
        """定义一个前向传播函数"""
        #mol_out是输入的节点特征矩阵，维度是节点数（N）*in_features
        #adj是输入的邻接矩阵，大小是N*N，表示节点之间的连接关系
        #对输入的特征矩阵进行线性变换，得到变换后的特征矩阵，大小为节点数*out_feautres
        #将每个节点的维度从in_features转变成out_features，等于nn.Linear()层
        atom_feature = torch.mm(mole_out, self.W) 
        N = atom_feature.size()[0]

        #计算单头注意力权重
        #假设一个节点与其他所有节点都相连的情况下，Whi||Whj得到的结果(N*N,2*out_features)
        # .repeat(1,N)在第一维上复制N次
        #atom_feature.repeat(1, N)得到的大小是(N,out_features*N),.view(N * N, -1)得到大小是(N*N,out_features)
        #atom_feature.repeat(N, 1)在第0维度上复制N次，得到的大小是(N*N,out_features)，torch.cat()得到的大小是(N*N,out_features*2)
        #最终得到大小为(N,N,2*out_features)的矩阵atom_trans，N是节点数目，每个节点与邻居节点的拼接矩阵(N,2*out_features)
        atom_trans = torch.cat([atom_feature.repeat(1, N).view(N * N, -1), atom_feature.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        #将拼接矩阵进行线性变换，得到大小为(N,N,1)的矩阵，.squeeze(2)表示去除索引为2的维度，即得到大小为(N,N)的矩阵
        e = self.leakyrelu(torch.matmul(atom_trans, self.a).squeeze(2)) 

        zero_vec = -9e15*torch.ones_like(e) #zero_vec的大小是N*N的矩阵，值为-9e15
        #如果两个节点之间有连接，则保留'e'中对应位置的值，否则对应位置的注意力权重将被替换为一个非常小的负数 -9e15
        attention = torch.where(adj > 0, e, zero_vec)

        #将注意力权重的信息保存在 atts_out 列表中，以便之后进行其他处理或可视化
        if self.inter_graph is not None:
            att_out = attention
            #检查 att_out 是否在 GPU 上，如果是，将其转移到 CPU 上
            if att_out.is_cuda:
                att_out = att_out.cpu()
            att_out = np.array(att_out)
            #将小于 -10000 的元素设置为 0
            att_out[att_out<-10000] = 0
            att_out = att_out.tolist()
            atts_out.append(att_out)
        
        attention = nn.functional.softmax(attention, dim=1) #确保权重的范围在[0,1]之间
        attention = self.dropout(attention) #dropout 用于减少注意力权重的过拟合
        output = torch.matmul(attention, atom_feature) #将 dropout 后的注意力权重 attention 与原子特征 atom_feature 相乘，得到更新后的节点特征

        if self.concat:
            return nn.functional.elu(output)
        else:
            return output 


class GATOne(nn.Module):
    """定义一个多头的GAT层（包含输入层-隐藏层-输出层），如果是单个的GAT分类模型，该函数输出即可得到分类概率"""
    def __init__(self,args):
        super(GATOne, self).__init__()
        self.nfeat = get_atom_features_dim() #节点的特征维度（原子）
        self.nhid = args.nhid #GAT隐藏层的维度
        self.dropout_gnn = args.dropout_gat # GAT 层中的 dropout 概率
        self.atom_dim = args.hidden_size #最终输出的原子特征的维度
        self.alpha = 0.2  # LeakyReLU 激活函数的负斜率参数
        self.nheads = args.nheads #GAT 层中注意力头的数量
        self.args = args
        self.dropout = nn.Dropout(p=self.dropout_gnn) #Dropout层
        
        if hasattr(args,'inter_graph'):
            self.inter_graph = args.inter_graph
        else:
            self.inter_graph = None

        #创建一个列表，内部有多个GAL层
        self.attentions = [GATLayer(self.nfeat, self.nhid, dropout_gnn=self.dropout_gnn, alpha=self.alpha, inter_graph=self.inter_graph, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        #将每个多头注意力实例 attention 添加到 GATOne 类的子模块中，并为它们分配一个名称 'attention_i'，其中 i 是对应的索引

        #创建了输出层 self.out_att，用于处理多头注意力层的输出并生成最终的输出结果
        self.out_att = GATLayer(self.nhid * self.nheads, self.atom_dim, dropout_gnn=self.dropout_gnn, alpha=self.alpha, inter_graph=self.inter_graph, concat=False)

    def forward(self,mole_out,adj):
        """定义一个前向传播函数，mole_out是表示分子图节点矩阵，adj是邻接矩阵表示原子之间的连接"""
        mole_out = self.dropout(mole_out) #一个dropout层
        #一个多头注意力的GAT层，得到隐藏层的结果
        mole_out = torch.cat([att(mole_out, adj) for att in self.attentions], dim=1) #得到多头注意力层的输出结果，使用拼接来整合结果
        #维度大小(N,out_features*n_heads)
        mole_out = self.dropout(mole_out)
        mole_out = nn.functional.elu(self.out_att(mole_out, adj))#经过一个GAT层，得到大小为N*self_atom_dim
        return nn.functional.log_softmax(mole_out, dim=1) #得到最终的模型输出结果。
        # 通常在训练过程中，会将该输出结果与真实标签进行比较，计算损失函数，并进行反向传播来更新模型的参数，以便优化模型的性能

class GATEncoder(nn.Module):
    """对批量的分子图进行特征提取"""
    def __init__(self,args):
        """定义一些参数和层"""
        super(GATEncoder,self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GATOne(self.args) #定义一个多头的GAT层
    
    def forward(self,mols,smiles):
        """定义一个前向传播神经网络,mols包含了一批分子图数据，smiles是包含多个smiles式的列表"""
        atom_feature, atom_index = mols.get_feature()
        #使用.get_feature()方法获得节点特征（atom_feature）以及每个分子图中节点的起始位置和节点数量（atom_index）
        if self.cuda:
            atom_feature = atom_feature.cuda()
        
        gat_outs=[]
        for i,one in enumerate(smiles):
            adj = []
            mol = Chem.MolFromSmiles(one)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol) #获取分子的邻接矩阵
            adj = adj/1 #为了确保邻接矩阵的数据类型为浮点数类型
            adj = torch.from_numpy(adj)
            if self.cuda:
                adj = adj.cuda()
            
            atom_start, atom_size = atom_index[i] #atom_index 是一个包含元组的列表，元组的个数等于分子图的数目
            #元组中包含了当前分子图中节点的起始位置以及当前分子图的节点个数
            one_feature = atom_feature[atom_start:atom_start+atom_size] #得到当前分子的原子特征向量
            
            gat_atoms_out = self.encoder(one_feature,adj) #每个原子在 GAT 层中的输出结果

            gat_out = gat_atoms_out.sum(dim=0)/atom_size #一个分子图的多头注意力层的输出结果进行池化操作
            gat_outs.append(gat_out)
        gat_outs = torch.stack(gat_outs, dim=0) #gat_outs 就包含了所有输入分子的 GAT 层输出。每一行对应一个分子的 GAT 层输出
        return gat_outs

class GAT(nn.Module):
    """一个完整的GAT模型的封装"""
    def __init__(self,args):
        super(GAT,self).__init__()
        self.args = args
        self.encoder = GATEncoder(self.args) #定义一个GAT层
        
    def forward(self,smile):
        """定义一个前向传播网络,smile包含一批smiles数据"""
        mol = create_graph(smile, self.args) #mol包含一批分子图数据
        gat_out = self.encoder.forward(mol,smile) #返回所有输入分子的全局特征

        return gat_out

class FpgnnModel(nn.Module):
    """整合FPN和GAT层"""
    def __init__(self,is_classif,gat_scale,cuda,dropout_fpn):
        """is_classif: 一个布尔值，表示模型是否用于分类任务,
        gat_scale: 一个浮点数,它的取值范围为[0, 1]，其中0表示只使用FPN编码器输出，1表示只使用GAT编码器输出，中间的值表示同时使用两者。
        self.sigmoid: 如果 is_classif 为True，将会初始化一个Sigmoid函数"""
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale #控制FPN和GAT合并的方式
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self,args):
        """定义一个GAT层"""
        self.encoder3 = GAT(args)
    
    def create_fpn(self,args):
        """定义一个FPN层"""
        self.encoder2 = FPN(args)
    
    def create_scale(self,args):
        """根据 gat_scale 的值选择创建不同规模的全连接层（Linear layer），并定义激活函数"""
        linear_dim = int(args.hidden_size) #控制编码器的输出维度
        if self.gat_scale == 1: #如果 gat_scale 为1，表示只使用 GAT 编码器的输出
            self.fc_gat = nn.Linear(linear_dim,linear_dim)
        elif self.gat_scale == 0:#如果 gat_scale 为0，表示只使用 FPN 编码器的输出
            self.fc_fpn = nn.Linear(linear_dim,linear_dim)
        else: #如果get_scale的值在[0,1]之间，同时使用 GAT 编码器和 FPN 编码器的输出
            self.gat_dim = int((linear_dim*2*self.gat_scale)//1)
            self.fc_gat = nn.Linear(linear_dim,self.gat_dim)
            self.fc_fpn = nn.Linear(linear_dim,linear_dim*2-self.gat_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self,args):
        """根据 gat_scale 的值创建一个用于分类任务的前馈神经网络"""
        linear_dim = args.hidden_size
        if self.gat_scale == 1:
            #如果 gat_scale 为1，表示只使用 GAT 编码器的输出。创建一个前馈神经网络 self.ffn
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     ) #args.task_num，用于分类任务
        elif self.gat_scale == 0:
            #如果 gat_scale 为0，表示只使用 FPN 编码器的输出。同样创建一个前馈神经网络 self.ffn
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )

        else:
            #否则，同时使用 GAT 编码器和 FPN 编码器输出。创建一个前馈神经网络 self.ffn
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim*2, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )
    
    def forward(self,input):
        """定义一个前向传播神经网络"""
        if self.gat_scale == 1: #如果 gat_scale 为1，表示只使用 GAT 编码器的输出
            output = self.encoder3(input)
        elif self.gat_scale == 0: #如果 gat_scale 为0，表示只使用 FPN 编码器的输出
            output = self.encoder2(input)
        else: #表示同时使用 GAT 编码器和 FPN 编码器的输出
            gat_out = self.encoder3(input) #输出维度就是linear_dim
            fpn_out = self.encoder2(input)
            gat_out = self.fc_gat(gat_out)
            gat_out = self.act_func(gat_out)
            
            fpn_out = self.fc_fpn(fpn_out)
            fpn_out = self.act_func(fpn_out)
            
            output = torch.cat([gat_out,fpn_out],axis=1)
        output = self.ffn(output) #进行分类任务的前向传播
        
        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        
        return output #函数返回经过分类任务前向传播后的输出结果

def get_atts_out():
    """存储每个图的注意力矩阵"""
    return atts_out

def FPGNN(args):
    """用于创建FPGNN模型"""
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = FpgnnModel(is_classif,args.gat_scale,args.cuda,args.dropout) #创建FP-GNN模型
    if args.gat_scale == 1:
        model.create_gat(args) #创建GAT编码器
        model.create_ffn(args) #只使用GAT编码器的输出结果创建前向传播神经网络
    elif args.gat_scale == 0:
        model.create_fpn(args) #创建FPN编码器
        model.create_ffn(args) #只使用FPN编码器的输出结果创建前向传播神经网络
    else:
        model.create_gat(args) #创建GAT编码器
        model.create_fpn(args) #创建FPN编码器
        model.create_scale(args) #对GAT和FPN编码器的输出结果进行缩放
        model.create_ffn(args) #使用两者结合的输出结果创建前向传播神经网络

    #对模型参数进行初始化
    for param in model.parameters():
        if param.dim() == 1: #对于维度为1的参数，使用零值初始化
            nn.init.constant_(param, 0)
        else: #对于维度大于1的参数，使用 Xavier 正态分布初始化
            nn.init.xavier_normal_(param)
    
    return model