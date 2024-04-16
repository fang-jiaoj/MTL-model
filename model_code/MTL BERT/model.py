import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PostionalEncoding(nn.Module):
    """该函数用于制作位置编码"""
    def __init__(self, d_model, max_len, device):
        super(PostionalEncoding, self).__init__()
        #max_len 是SMILES的最大长度，d_model 是元素/字符的向量维度。
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False #这个位置编码张量被设为不需要梯度计算，因为位置编码不会随着训练而改变。

        pos = torch.arange(0, max_len, device=device) #从 0 到 max_len-1 的整数，表示每个元素/字符的位置
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        ##_2i 包含了从 0 到 d_model-1 的整数，以 2 为步长递增。这两个张量将用于计算正弦和余弦函数的输入。

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        #对于位置编码张量的偶数列（0-based 索引），它们由正弦函数计算，而奇数列则由余弦函数计算，最终的维度是（max_len,d_model）

    def forward(self, x):
        batch_size, seq_len = x.size() #得到SMILES式的个数以及每个SMILES的长度
        return self.encoding[:seq_len, :].unsqueeze(0)
        #.unsqueeze(0) 它在张量的维度之前插入一个新的维度，它将 (seq_len, d_model) 形状的张量转换为 (1, seq_len, d_model) 形状的张量，
        # 这是因为 Transformer 模型期望输入数据的维度是 (batch_size, seq_len, d_model)，其中 batch_size 表示批次大小，通常为 1

def make_src_mask(src,src_pad_idx=0):
    """用于创建源序列掩码的函数"""
    src_mask = (src == src_pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    #.unsqueeze(1) 将这个布尔张量的形状从 (batch_size, seq_len) 变为 (batch_size, 1, seq_len)，其中 batch_size 表示批次大小，seq_len 表示序列的长度。这是为了与后续的操作相兼容。

    #.unsqueeze(2) 进一步将形状变为 (batch_size, 1, 1, seq_len)，添加了一个额外的维度，以便与后续的注意力操作相兼容。在 Transformer 模型中，这个维度通常表示注意力头数，但在这里仅为了维度匹配而添加。
    return src_mask #返回的 src_mask 张量是一个掩码张量，其中填充位置被标记为 True，其他位置被标记为 False，用于在自注意力机制中遮蔽填充位置

def make_trg_mask(trg,trg_pad_idx=0):
    """用于创建目标序列掩码的函数,
    在解码过程中防止模型查看未来时刻的信息以及在填充位置上分配注意力权重？？？？"""
    trg_pad_mask = (trg == trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
    trg_mask = trg_pad_mask & trg_sub_mask
    #将填充位置掩码与下三角掩码相结合，以得到最终的目标序列掩码。这个掩码将在模型的注意力计算中使用，以确保模型不会在填充位置和未来时刻的位置上分配注意力权重。
    return trg_mask.to(device)

class ScaledDotProductAttention(nn.Module):
    """实现缩放点积注意力机制（Scaled Dot-Product Attention）的 PyTorch 模块"""
    def __init__(self,dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k],batch_size 表示批次大小，n_heads 表示注意力头的数量，len_q 表示查询SMILES的长度，d_k 表示每个元素/字符的维度
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.shape[-1])  # scores : [batch_size, n_heads, len_q, len_k]
        #计算查询张量 Q 和键张量 K 之间的点积注意力分数

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
            #将掩码中的 True 位置（即应该被屏蔽的位置）的分数设置为一个很大的负数，这样在 softmax 操作中它们将接近零

        attn = F.softmax(scores,dim=-1) #将分数转换为注意力权重
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn #最后可得到融合上下文信息的向量编码，维度与输入维度相同和注意力权重矩阵


class MultiHeadAttention(nn.Module):
    """多头注意力机制（Multi-Head Attention）的 PyTorch 模块,
    d_model 是输入特征的维度。
    num_heads 是注意力头的数量。
    rate 是用于 dropout 操作的丢弃概率。"""
    def __init__(self,d_model,num_heads,rate):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0 #检查 d_model 是否可以被 num_heads 整除，以确保能够均匀分配特征到各个注意力头

        self.depth = d_model // self.num_heads #得到每个头分得的元素数

        #初始化了多个线性层（self.W_Q, self.W_K, self.W_V 和 self.fc）来映射输入特征到查询（Q）、键（K）、值（V）以及输出
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.layernorm = nn.LayerNorm(d_model) #创建了一个 Layer Normalization 层（self.layernorm）用于归一化输出
        self.dropout = nn.Dropout(p=rate) #创建了一个 Dropout 层（self.dropout）用于正则化
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate=rate) #Self-Attention层

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        #input_Q,input_K,input_V均指的是输入模型的X（数值化的SMILES+位置embedding）
        residual, batch_size = input_Q, input_Q.size(0) #方便进行残差连接
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        #将语义空间划分成多个子空间，在不同的空间中Attention关注不同的部分，生成多个Self-Attention的Q,K,V
        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        #首先将输入的X进行线性变换得到Q，维度是（batch_size,len_q,d_model）-->(batch_size,len_q,n_head,depth)-->(batch_size,n_head,len_q,depth)
        K = self.W_K(input_K).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.num_heads, self.depth).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.dot_product_attention(Q, K, V, attn_mask) #得到融合上下文信息的向量矩阵和注意力权重矩阵，多个头就得到多个向量矩阵
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        # context: [batch_size, len_q, n_heads * d_v]，得到融合多个头信息的向量矩阵
        output = self.fc(context) # [batch_size, len_q, d_model]，将融合后的输出矩阵进行线性变换，与输入的维度一致
        output = self.dropout(output) #经过dropout层，防止过拟合
        return self.layernorm(output + residual), attn
        #将输出与输入的残差连接起来，并通过 Layer Normalization 进行处理
        # 最终的输出结果是多头自注意力操作的结果和注意力矩阵。这个过程会在模型的不同位置多次重复，以捕获输入序列的不同关系和特征

class PoswiseFeedForwardNet(nn.Module):
    """用于多头自注意力模型中的前馈神经网络 (FeedForward Network) 的模块"""
    def __init__(self,d_model,dff, rate):
        #d_model: 输入和输出的特征维度。
        #dff: 前馈神经网络中隐藏层的维度。
        #rate: 丢弃率，用于在训练过程中进行丢弃操作
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Linear(dff, d_model)
        ) #定义一个前馈神经网络
        self.dropout = nn.Dropout(p=rate) #定义一个dropout层
        self.layernorm = nn.LayerNorm(d_model) #定义一个归一化层
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]，输入是多头注意力层的输出
        '''
        residual = inputs #保存输入 inputs 作为残差连接的一部分
        output = self.fc(inputs) #通过前馈神经网络 self.fc 对输入进行处理，得到输出
        output = self.dropout(output) #应用了一个丢弃操作 self.dropout，以减少过拟合
        return self.layernorm(output + residual) #最后进行残差连接和归一化

class EncoderLayer(nn.Module):
    """这是一个 Transformer 模型中的编码器层（Encoder Layer）模块，将上面的多头注意力层和前馈神经网络层连接起来"""
    def __init__(self,d_model, num_heads, dff, rate=0.1):
        #d_model: 输入和输出的特征维度。
        #num_heads: 多头自注意力机制中的头数。
        #dff: 前馈神经网络中隐藏层的维度。
        #rate: 丢弃率，用于在训练过程中进行丢弃操作
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model,num_heads,rate) #一个多头注意力层
        self.pos_ffn = PoswiseFeedForwardNet(d_model,dff,rate) #一个前馈神经网络层

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        #先通过多头注意机制得到融合上下文信息的向量表示和注意力矩阵，注意力矩阵记录了输入序列中不同位置之间的关联程度
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        #再输入前馈神经网络中，再经过残差连接和归一化，得到最终的编码器层输出
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn #模型对输入序列的编码表，可以用于后续任务，注意力矩阵 attn 可用于分析模型对输入序列的关注程度


class Encoder(nn.Module):
    """预训练部分的模型框架"""
    def __init__(self, num_layers,d_model,num_heads,dff,input_vocab_size,maximum_position_encoding=2000,rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model #词嵌入的维度
        self.num_layers = num_layers #encoder的层数
        self.src_emb = nn.Embedding(input_vocab_size, d_model) #词嵌入层，用于将输入SMILES中的数值化的元素转换为词嵌入向量（d_model）
        self.pos_emb = PostionalEncoding(d_model,2000,device=device) #位置编码层
        self.layers = nn.ModuleList([EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)])
        #由多个编码器层（EncoderLayer）组成的模块列表，每个编码器层包含自注意力机制和前馈神经网络
        self.dropout = nn.Dropout(rate) #Dropout层

    def forward(self, enc_inputs):

        # nn.TransformerEncoder
        '''
        enc_inputs: [batch_size, src_len]，指的是数值化的SMILES序列
        '''
        word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]，将输入序列 enc_inputs 转换为词嵌入 word_emb
        pos_emb = self.pos_emb(enc_inputs) # [batch_size, src_len, d_model]，引入位置编码
        enc_outputs = word_emb + pos_emb #获得包含位置信息的输入表示 enc_outputs
        enc_self_attn_mask = make_src_mask(enc_inputs) #[batch_size,1,1 src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask) #通过多个编码器层 layers 逐层处理 enc_outputs
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns #获得多个encoder编码后的元素表示 enc_outputs 和每个encoder层的注意力矩阵列表 enc_self_attns

class EncoderForPrediction(nn.Module):
    """微调部分的模型框架"""
    def __init__(self, num_layers,d_model,num_heads,dff,input_vocab_size,maximum_position_encoding=2000,rate=0.1,prediction_nums=0):
        super(EncoderForPrediction, self).__init__()

        self.d_model = d_model #元素/符号对应的维度
        self.num_layers = num_layers #encoder的层数
        self.num_heads = num_heads #Self-Attention的数目
        self.prediction_nums = prediction_nums #进行的任务数
        self.src_emb = nn.Embedding(input_vocab_size, d_model) #词嵌入层，将数值化的SMIELS转换为词嵌入。
        self.pos_emb = PostionalEncoding(d_model,2000,device=device) #位置嵌入层，为每个位置添加位置信息
        self.layers = nn.ModuleList([EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)])
        #一个包含多个 EncoderLayer 层的列表
        self.dropout = nn.Dropout(rate) #nn.Dropout 层用于在模型的训练过程中进行随机丢弃，以减少过拟合

    def forward(self, enc_inputs):

        # nn.TransformerEncoder
        '''
        enc_inputs: [batch_size, src_len]，指的是数值化的SMIELS序列
        '''
        word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]，将输入数值化的SMILES转换为词嵌入（word_emb）
        pos_emb = self.pos_emb(enc_inputs[:,self.prediction_nums:]) # [batch_size, src_len, d_model]
        #只为元素SMILES字符的元素部分添加位置编码，任务token不添加位置编码，保持与预训练部分一致

        enc_outputs = word_emb

        enc_outputs[:,self.prediction_nums:] += pos_emb
        #将词嵌入和位置编码相加，得到编码器的输入,但不包括任务token，输入部分：不加位置编码的任务嵌入和加位置编码的词嵌入

        enc_self_attn_mask = make_src_mask(enc_inputs) # [batch_size,1,1 src_len]

        enc_self_attn_mask =  enc_self_attn_mask.repeat(1, self.num_heads, enc_self_attn_mask.shape[-1], 1)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        #将原始的二维掩码 enc_self_attn_mask 复制多次以匹配多头注意力的数量，
        # 以便每个头都可以使用相同的掩码进行自注意力计算。这是多头注意力的一部分，确保每个头都能够关注到合适的信息而不会受到未来位置的干扰
        # enc_self_attn_mask[:,:,:,:self.prediction_nums]=1
        # enc_self_attn_mask[:, :, torch.arange(self.prediction_nums), torch.arange(self.prediction_nums)] = 0

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns #返回编码器的输出和自注意力权重列表


class BertModel(nn.Module):
    """该类定义了一个Bert预训练模型，返回掩蔽词属于字典中每个词的概率"""
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 50,dropout_rate = 0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=2000,rate=dropout_rate)
        #采用了6个EncoderLayer，用于处理输入数据，特征提取层
        #两个线性层，用于将Transformer编码器的输出映射到最终的输出词汇大小，作为任务层
        self.fc1 = nn.Linear(d_model,d_model*2)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model*2,vocab_size)


    def forward(self,x):
        """x是数值化的SMILES列表，维度（batch_size,src_len）"""
        x,attns = self.encoder(x) #通过Transformer编码器处理输入数据x，返回融合上下文信息的向量表示，维度（batch_size,src_len,d_model）
        #attns是6个注意力矩阵列表，维度是（batch_size,num_heads,src_len,src_len）
        y = self.fc1(x)
        y = self.dropout1(y)
        y = F.gelu(y)
        y = self.fc2(y) #最终生成掩蔽的词属于字典中每个词的概率，维度（batch_size，src_len,vocab_size）
        return y


class PredictionModel(nn.Module):
    """该类定义了一个微调模型，同时进行多个回归（regression）和分类（classification）任务"""
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 60,dropout_rate = 0.1, reg_nums=0,clf_nums=0):
        super(PredictionModel, self).__init__()

        self.reg_nums = reg_nums #回归任务数
        self.clf_nums = clf_nums #分类任务数

        self.encoder = EncoderForPrediction(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,
                               maximum_position_encoding=2000,rate=dropout_rate,
                               prediction_nums=self.reg_nums+self.clf_nums)
        #EncoderForPrediction 是微调部分的模型框架，接受输入并生成编码后的表示



        self.fc_list = nn.ModuleList()

        #这是一个由多个线性层组成的模块列表，每个线性层都用于不同的任务。在这个列表中，每个任务都有两个线性层，其目的是将编码的表示映射到最终的输出
        for i in range(self.clf_nums+self.reg_nums):
            self.fc_list.append(nn.Sequential(nn.Linear(d_model,2*d_model),
                                              nn.Dropout(0.1),nn.LeakyReLU(0.1),
                                              nn.Linear(2*d_model,1)))

        # self.fc1 = nn.Linear(d_model,2*d_model)
        # self.dropout1 = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(2*d_model,1)

    def forward(self,x):
        """x是数值化后的SMILES序列"""
        x,attns = self.encoder(x) #返回融合上下文信息的向量表示（batch_size,src_len,d_model）和注意力权重矩阵列表（batch_size,n_heads,src_len,src_len）

        ys = []

        #通过 fc_list 中的线性层生成每个任务的预测结果
        for i in range(self.clf_nums+self.reg_nums):
            y = self.fc_list[i](x[:,i]) #将任务token对应的向量编码送入全连接层，得到预测结果
            ys.append(y)

        y = torch.cat(ys,dim=-1)  #y的维度（batch_size,clf_nums+reg_nums）

        # y = self.fc1(x)
        # y = self.dropout1(y)
        # y = self.fc2(y)
        # y = y.squeeze(-1)
        properties = {'clf':None,'reg':None}
        if self.clf_nums>0:
            clf = y[:,:self.clf_nums] #分类任务的预测标签
            properties['clf'] = clf
        if self.reg_nums>0:
            reg = y[:,self.clf_nums:] #回归任务的预测值
            properties['reg'] = reg
        return properties #该字典中包括了分类任务的结果和回归任务的结果

