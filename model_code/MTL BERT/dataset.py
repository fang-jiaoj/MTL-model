import pandas as pd
import numpy as np
import torch
import rdkit
from rdkit import Chem
import re

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #指定设备

smiles_regex_pattern = r'Si|Mg|Ca|Fe|As|Al|Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]icosn]|/|\\'
#该正则表达式模式用于匹配SMILES表示中的化学元素和特定字符，这个模式可以用来检测SMILES字符串中是否包含这些元素和字符

smiles_str2num = {'<PAD>':0 ,'Cl': 1, 'Br': 2, '#': 3, '(': 4, ')': 5, '+': 6, '-': 7, '0': 8, '1':  9,
    '2': 10,'3': 11,'4':12,'5':13,'6':14,'7':15,'8':16,'9':17,':':18,'=':19,'@':20,'C':21,
    'B':22,'F':23,'I':24,'H':25,'O':26,'N':27,'P':28,'S':29,'[':30,']':31,'c':32,'i':33,'o':34,
    'Si': 35, 'Mg': 36, 'Ca': 37, 'Fe': 38, 'As': 39, 'Al': 40,
    'n':41,'p':42,'s':43,'%':44,'/':45,'\\':46,'<MASK>':47,'<UNK>':48,'<GLOBAL>':49,'<p1>':50,
    '<p2>':51,'<p3>':52,'<p4>':53,'<p5>':54,'<p6>':55,'<p7>':56,'<p8>':57,'<p9>':58,'<p10>':59}
#将每个SMILES字符映射到一个数字(编码)，'<PAD>'、'<MASK>'、'<UNK>'等，用于特殊情况的处理，比如填充、掩码和未知字符

smiles_num2str =  {i:j for j,i in smiles_str2num.items()} #将数字映射为字符，相当于解码

smiles_char_dict = list(smiles_str2num.keys()) #字符列表

def randomize_smile(sml):
    """执行SMILES枚举策略"""
    m = Chem.MolFromSmiles(sml)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans) #对原子索引列表进行随机重排
    nm = Chem.RenumberAtoms(m, ans) #使用重排后的原子索引列表来重新编号分子中的原子
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles

def canonical_smile(sml):
    """对分子进行标准化处理"""
    m = Chem.MolFromSmiles(sml)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles

class Smiles_Bert_Dataset(Dataset):
    #将原始 SMILES 数据转换为数值化的特征序列并且进行MASK操作，预训练时使用的掩码训练操作
    def __init__(self, path, Smiles_head):
        if path.endswith('txt'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)

        #初始化数据和词汇表
        self.data = self.df[Smiles_head].to_numpy().reshape(-1).tolist() #是一个smiles列表
        self.vocab = smiles_str2num #相当于编码器
        self.devocab = smiles_num2str  #相当于解码器

    def __len__(self):
        return len(self.data) #数据集的长度

    def __getitem__(self, item):
        smiles = self.data[item]
        x, y, weights = self.numerical_smiles(smiles)
        return x, y, weights #返回经过修改的序列，原始序列以及哪些位置进行了修改

    def numerical_smiles(self, smiles):
        """这段代码主要是进行了数据增强操作，对数值化的 SMILES 序列进行了一些随机的字符替换和位置加权，以模拟数据的多样性"""
        nums_list = self._char_to_idx(smiles) #将输入的 SMILES 字符串转换为数值化的序列

        choices = np.random.permutation(len(nums_list) - 1)[:int(len(nums_list) * 0.15)] + 1
        #对序列索引进行随机排列，生成一个随机的索引顺序，选择随机排列后的索引序列的前 15% 部分（masked），以使索引从 1 开始
        y = np.array(nums_list).astype('int64') #y就表示了原始的数值化 SMILES 序列
        weight = np.zeros(len(nums_list)) #初始化权重
        for i in choices:
            rand = np.random.rand() #生成[0, 1) 范围内的随机值
            weight[i] = 1 #被masked后的位置的权重设为1
            if rand < 0.8: #即以 0.8 的概率执行下面的操作
                nums_list[i] = smiles_str2num['<MASK>'] #这个操作以较高的概率使得模型在这个位置上预测为 <UNK>
            elif rand < 0.9: #以 0.1 的概率执行下面的操作
                nums_list[i] = int(np.random.rand() * 46 + 0.1)
                #将该位置的值修改为0到46之间的随机整数,这个操作以较低的概率使得模型在这个位置上被其他随机的字符替换

        x = np.array(nums_list).astype('int64') #进行MASK操作后的数值化 SMILES 序列
        weights = weight.astype('float32') #这个权重数组记录了哪些位置进行了MASK操作
        return x, y, weights #修改后的 x 表示经过MASK后的序列，而 y 表示原始序列，weights 则表示哪些位置进行了MASK操作

    def _char_to_idx(self, seq):
        """将一个 SMILES 字符串转换为一个数字序列，其中每个数字表示相应的化学元素或特殊符号在词汇表中的编码"""
        char_list = re.findall(smiles_regex_pattern, seq)
        #使用正则表达式（smiles_regex_pattern）从输入的SMILES字符串（seq）中提取出匹配的化学元素和特殊符号，生成一个包含SMILES字符串中元素和特殊符号的列表
        #从输入的 SMILES 字符串(seq)中提取出匹配的化学元素和特殊符号。例如，它可以识别出 'C', 'N', 'O', 'Cl' 等元素
        char_list = ['<GLOBAL>'] + char_list
        return [self.vocab.get(char_list[j],self.vocab['<UNK>']) for j in range(len(char_list))]
        #这个方法返回一个数字序列，其中每个数字表示SMILES字符串中的一个元素或特殊符号在词汇表中的编码


class Prediction_Dataset(object):
    """将smiles处理成适合微调模型的输入，包括任务token：<p1>,<p2>,<p3>...加上将SMILES变成数字的序列"""
    def __init__(self, df, smiles_head='SMILES', reg_heads=[],clf_heads=[]):
        """reg_heads: 这是一个列表，包含了 DataFrame 中包含数值属性的列的名称,clf_heads: 这是一个列表，包含了 DataFrame 中包含分类属性的列的名称"""
        self.df = df
        self.reg_heads = reg_heads
        self.clf_heads = clf_heads

        self.smiles = self.df[smiles_head].to_numpy().reshape(-1).tolist()

        self.reg = np.array(self.df[reg_heads].fillna(-1000)).astype('float32')
        #DataFrame 中提取分类属性列的数据，用 -1000 填充缺失值
        self.clf = np.array(self.df[clf_heads].fillna(-1000)).astype('int32')

        self.vocab = smiles_str2num
        self.devocab = smiles_num2str

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        smiles = self.smiles[item]

        properties = [None, None] #用于存储分类标签和回归标签的值
        if len(self.clf_heads)>0:
            clf = self.clf[item]
            properties[0] = clf # 将获取的分类属性值存储在 properties 列表的第一个位置

        if len(self.reg_heads)>0:
            reg = self.reg[item]
            properties[1] = reg #获取的数值属性值存储在 properties 列表的第二个位置

        nums_list = self._char_to_idx(seq=smiles) #将 SMILES 字符串中的每个字符转换为相应的数字编码
        if len(self.reg_heads) + len(self.clf_heads) >0: #查看任务数，加上任务token
            ps = ['<p{}>'.format(i+1) for i in range(len(self.reg_heads) + len(self.clf_heads))]
            nums_list = [smiles_str2num[p] for p in ps] + nums_list
            #将特殊标记的数字编码添加到 nums_list 列表的开头，相当于将任务1-5所对应的数字和SMILES对应的数字编码作为微调阶段的输入
        x = np.array(nums_list).astype('int32')
        return x, properties #返回输入微调模型的token以及对应的标签列表

    def numerical_smiles(self, smiles):
        smiles = self._char_to_idx(seq=smiles)
        x = np.array(smiles).astype('int64')
        return x


    def _char_to_idx(self, seq):
        """将SMILES变成数值序列"""
        char_list = re.findall(smiles_regex_pattern, seq)
        char_list = ['GLOBAL'] + char_list
        return [self.vocab.get(char_list[j],self.vocab['<UNK>']) for j in range(len(char_list))]

class Pretrain_Collater():
    """返回整理后的批次数据，包括转换后的 xs、ys 和 weights 张量，这个对应预训练阶段的输入"""
    def __init__(self):
        super(Pretrain_Collater, self).__init__()
    def __call__(self,data):
        xs, ys, weights = zip(*data) #各个数据项的维度解压成三个分开的列表

        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        #使用 pad_sequence 函数将它们填充成一个批次，xs 和 ys 需要被填充成相同的长度，以便进行模型训练
        # 并将结果张量转换为 long 数据类型并移动到指定的设备（通常是 GPU）上
        ys = pad_sequence([torch.from_numpy(np.array(y)) for y in ys], batch_first=True).long().to(device)
        weights = pad_sequence([torch.from_numpy(np.array(weight)) for weight in weights], batch_first=True).float().to(device)

        return xs, ys, weights


class Finetune_Collater():
    def __init__(self,args):
        super(Finetune_Collater, self).__init__()
        self.clf_heads = args.clf_heads

    def __call__(self, data):
        xs, properties_list = zip(*data) #xs 包含所有的 x 数据，properties_list 包含所有的属性数据
        xs = pad_sequence([torch.from_numpy(np.array(x)) for x in xs], batch_first=True).long().to(device)
        #使用 pad_sequence 函数将它们填充成一个批次，并将结果张量转换为 long 数据类型并移动到指定的设备（通常是 GPU）上
        properties_dict = {'clf':None,'reg':None}

        if len(self.clf_heads) >0:
           properties_dict['clf'] = torch.from_numpy(np.concatenate([p[0].reshape(1,-1) for p in properties_list],0).astype('int32')).to(device)
            #np.concatenate()可以用来将多个数组按照指定的轴进行连接，生成一个新的数组，可以生成一个标签张量（样本数*标签数）
        return xs, properties_dict #整理后的批次数据，包括转换后的 xs 张量和标签字典 properties_dict，这个作为微调阶段的输入
