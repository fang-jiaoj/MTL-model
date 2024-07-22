import os
import csv
import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve, roc_auc_score
from fpgnn.data import MoleDataSet, MoleData, scaffold_split
from fpgnn.model import FPGNN
from rdkit import Chem

def mkdir(path,isdir = True):
    """该函数用于创建目录,path表示目录路径，isdir指示是否要创建目录（默认为 True）"""
    if isdir == False:
        path = os.path.dirname(path) #如果 isdir 为 False，则将 path 解析为文件路径，使用 os.path.dirname() 函数获取其所在的目录路径
    if path != '':
        os.makedirs(path, exist_ok = True) #如果 path 不为空（即指定的路径不是根目录），则使用 os.makedirs() 函数创建目录

def set_log(name,save_path):
    """设置日志记录器，它接受两个参数：name 表示日志记录器的名称，save_path 表示日志文件保存的路径"""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    
    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)
    log.addHandler(log_stream)
    
    mkdir(save_path)
    
    log_file_d = logging.FileHandler(os.path.join(save_path, 'debug.log'))
    log_file_d.setLevel(logging.DEBUG)
    log.addHandler(log_file_d)
    
    return log

def get_header(path):
    """函数用于获取CSV文件的头部信息，即文件的第一行内容（通常是列名）"""
    with open(path) as file:
        header = next(csv.reader(file)) #创建一个CSV文件的阅读器对象
    
    return header

def get_task_name(path):
    """函数用于获取CSV文件中的任务名称"""
    task_name = get_header(path)[1:]
    
    return task_name

def load_data(path,args):
    """该函数用于从CSV文件中加载数据并创建一个 MoleDataSet 对象，其中包含有效的分子数据"""
    with open(path) as file:
        reader = csv.reader(file) #使用 csv.reader 来读取CSV文件中的每一行数据
        next(reader) #跳过文件的表头
        lines = []
        for line in reader:
            lines.append(line)
        data = []
        for line in lines:
            one = MoleData(line,args) #创建一个MoleData对象
            data.append(one) #得到包含所有MoleData对象的列表
        data = MoleDataSet(data) #将所有 MoleData 对象封装为一个数据集对象

        fir_data_len = len(data)
        data_val = []
        smi_exist = []
        for i in range(fir_data_len):
            if data[i].mol is not None: #如果moleData对应的mol式不为空就是有效的
                smi_exist.append(i)
        data_val = MoleDataSet([data[i] for i in smi_exist]) #将有效的 MoleData 对象封装为一个新的数据集对象
        now_data_len = len(data_val)
        print('There are ',now_data_len,' smiles in total.')
        if fir_data_len - now_data_len > 0:
            print('There are ',fir_data_len , ' smiles first, but ',fir_data_len - now_data_len, ' smiles is invalid.  ')


    return data_val

def split_data(data,type,size,seed,log):
    """将数据集 data 划分为训练集、验证集和测试集，并根据指定的划分类型和大小进行划分"""
    #确保 size 是一个长度为 3 的列表，其中包含三个划分比例，这三个比例之和应为 1
    assert len(size) == 3
    assert sum(size) == 1

    #根据指定的划分类型 type 进行划分
    if type == 'random': #随机划分方法
        data.random_data(seed)
        train_size = int(size[0] * len(data))
        val_size = int(size[1] * len(data))
        train_val_size = train_size + val_size
        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]
    
        return MoleDataSet(train_data),MoleDataSet(val_data),MoleDataSet(test_data) #返回MoleDataSet 对象
    elif type == 'scaffold': #分子骨架划分方法（scaffold split）
        return scaffold_split(data,size,seed,log) #返回相应的 MoleDataSet 对象
    else:
        raise ValueError('Split_type is Error.')

def get_label_scaler(data):
    #对标签进行归一化处理，并返回标签的均值和标准差
    #获取data中所有的smiles式和标签
    smile = data.smile()
    label = data.label()
    
    label = np.array(label).astype(float)
    ave = np.nanmean(label,axis=0) #计算数组中非NaN值的均值，维度是(1,num_tasks)
    ave = np.where(np.isnan(ave),np.zeros(ave.shape),ave) #np.where(condition, x, y): 这是 NumPy 中的条件选择函数
    #对于ave数组中的NaN值，用0替换它们，而保留非NaN值不变
    std = np.nanstd(label,axis=0)
    std = np.where(np.isnan(std),np.ones(std.shape),std) #对于 std 数组中的NaN值，用1替换它们，而保留非NaN值不变
    std = np.where(std==0,np.ones(std.shape),std)
    # 对于 std 数组中的值为零的元素，用1替换它们，而保留非零值不变，最终的维度是(1,Num_tasks)

    change_1 = (label-ave)/std #对标签进行归一化
    label_changed = np.where(np.isnan(change_1),None,change_1) #对于change_1数组中的NaN值，用None替换它们，而保留非NaN值不变
    label_changed.tolist()
    data.change_label(label_changed) #更新标签
    
    return [ave,std] #维度是(1,num_tasks)

def get_loss(type):
    """返回适合不同任务类型的损失函数"""
    if type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none') #对于二分类任务，使用二元交叉熵损失函数
    #reduction='none' 意味着不进行损失的求和或平均操作，而是返回每个样本的单独损失值
    elif type == 'regression':
        return nn.MSELoss(reduction='none') #对于回归任务，使用均方误差损失函数
    else:
        raise ValueError('Data type Error.')

def prc_auc(label,pred):
    """用于计算 PR 曲线下的面积(PR-AUC)"""
    prec, recall, _ = precision_recall_curve(label,pred)
    result = auc(recall,prec)  #PRC-AUC 值用于评估模型在非平衡类别（类别不均衡）的二分类问题中的性能，尤其在正类样本较少的情况下
    return result

def rmse(label,pred):
    """计算MSE，常用的回归问题的性能评估指标"""
    result = mean_squared_error(label,pred)
    return math.sqrt(result)

def get_metric(metric):
    """选择不同的模型评估指标"""
    if metric == 'auc':
        return roc_auc_score
    elif metric == 'prc-auc':
        return prc_auc
    elif metric == 'rmse':
        return rmse
    else:
        raise ValueError('Metric Error.')

def save_model(path,model,scaler,args):
    """保存训练好的模型及其相关信息到文件中,
    path：要保存模型的文件路径;model：要保存的模型实例;scaler：用于标准化数据的 Scaler，可能为 None，表示没有数据标准化"""
    if scaler != None: #判断数据是否已经标准化，创建state 字典中包含了模型的状态字典（即模型的权重）
        state = {
            'args':args,
            'state_dict':model.state_dict(), #保存了模型的权重（参数）
            'data_scaler':{
                'means':scaler[0],
                'stds':scaler[1]
            }
        }
    else:
        state = {
            'args':args,
            'state_dict':model.state_dict(),
            'data_scaler':None
            }
    torch.save(state,path) #将 state 字典保存到指定的文件路径 path 中，可以使用torch.load()加载模型

def load_model(path,cuda,log=None,pred_args=None):
    #根据是否提供日志对象来选择使用不同的调试输出方式
    if log is not None:
        debug = log.debug
    else:
        debug = print
    
    state = torch.load(path,map_location=lambda storage, loc: storage) #将从指定路径path加载模型状态
    args = state['args'] #模型训练时使用的参数和配置信息

    #如果在加载模型时传入了 pred_args 参数，那么它会检查 pred_args 中的参数，并将其中没有出现在 args 中的参数添加到 args 中。这样可以确保加载的模型使用了正确的参数配置
    if pred_args is not None:
        for key,value in vars(pred_args).items():
            if not hasattr(args,key):
                setattr(args, key, value)
    
    state_dict = state['state_dict'] #包含了模型中所有层的参数张量
    
    model = FPGNN(args)
    model_state_dict = model.state_dict() #model.state_dict() 返回了当前模型 model 的状态字典，也就是包含了所有层的参数张量。这个状态字典是模型当前的权重和状态

    #在加载预训练的模型参数时进行参数匹配和加载
    #model_state_dict 表示当前模型的参数，state_dict 表示预训练模型保存的参数
    load_state_dict = {}
    for param in state_dict.keys():
        if param not in model_state_dict: #检查当前模型是否具有与预训练模型相同的参数名(param)
            debug(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != state_dict[param].shape: #检查当前模型中的参数形状是否与预训练模型的参数形状相同
            debug(f'Shape of parameter is error: {param}.')
        else: #如果参数名和形状都匹配，将预训练模型的参数加载到 load_state_dict 字典中
            load_state_dict[param] = state_dict[param]
            debug(f'Load parameter: {param}.')

    #将匹配成功的预训练模型参数 load_state_dict 更新到当前模型的参数字典 model_state_dict 中
    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict) #将更新后的参数加载到当前模型中
    
    if cuda:
        model = model.to(torch.device("cuda"))
    
    return model

def get_scaler(path):
    """用于从已保存的模型文件中获取数据标准化的均值和标准差"""
    state = torch.load(path, map_location=lambda storage, loc: storage)
    if state['data_scaler'] is not None:
        ave = state['data_scaler']['means']
        std = state['data_scaler']['stds']
        return [ave,std]
    else:
        return None

def load_args(path):
    """用于从已保存的模型文件中加载训练时所用的参数配置信息"""
    state = torch.load(path, map_location=lambda storage, loc: storage)
    
    return state['args']

def rmse(label,pred):
    """计算回归任务中的均方根误差,RMSE越小，表示模型的预测与真实值越接近，性能越好"""
    result = mean_squared_error(label,pred)
    result = math.sqrt(result)
    return result


"""

Noam learning rate scheduler with piecewise linear increase and exponential decay.

The learning rate increases linearly from init_lr to max_lr over the course of
the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
Then the learning rate decreases exponentially from max_lr to final_lr over the
course of the remaining total_steps - warmup_steps (where total_steps =
total_epochs * steps_per_epoch). This is roughly based on the learning rate
schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).

"""

class NoamLR(_LRScheduler):
    """该类是一个自定义的学习率调度器，它用于在训练过程中动态地调整学习率，以帮助模型更好地收敛到最优解"""
    def __init__(self,optimizer,warmup_epochs,total_epochs,steps_per_epoch,
                 init_lr,max_lr,final_lr):
        """optimizer:优化器，通常是PyTorch中的torch.optim.Optimizer的实例；Warmup阶段的epoch数，一个数组，每个元素对应不同的参数组的Warmup epoch数；
        total_epochs: 总的训练epoch数，一个数组，每个元素对应不同的参数组的总epoch数；steps_per_epoch: 每个epoch的训练步数（batch数）；
        init_lr: 初始学习率，一个数组；max_lr: 最大学习率，一个数组，在Warmup阶段结束后达到最大值；final_lr: 最终学习率，在Exponential Decay阶段最小值；
        """
        #Warmup阶段：在训练的前几个epoch中，学习率从初始值逐渐增加到最大值，
        # 这个过程称为Warmup。在Warmup阶段，学习率逐渐增加，有助于模型更快地找到一个合适的区域，并避免训练过程中梯度过大的情况。
        #Exponential Decay阶段：在Warmup之后，学习率按指数方式逐渐减小，使得模型能够更加细致地搜索最优解。
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr) #optimizer.param_groups是一个包含优化器参数组的列表，每个元素对应不同的参数组。为了确保这些参数都对应相同的参数组

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0 #记录当前训练的步骤数，初始值为0
        self.lr = init_lr
        #每个参数组对应的预热阶段的总步骤数，self.steps_per_epoch表示每个轮数中的训练步骤数
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        #每个参数组对应的总训练步骤数
        self.total_steps = self.total_epochs * self.steps_per_epoch
        #每个参数组在预热阶段学习率的线性增量
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        #计算每个参数组在预热阶段后学习率逐渐降低时的指数因子
        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """返回当前学习率的列表"""
        return list(self.lr)

    def step(self,current_step=None):
        """当前步数和预定义的参数，计算并更新学习率，并将其应用到优化器中的参数组上"""
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs): #遍历优化器中的参数组
            if self.current_step <= self.warmup_steps[i]: #如果当前步骤数小于预热阶段步骤数
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]: #如果当前步骤数大于预热阶段步骤数小于总步骤数
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i] #更新优化器中的参数组的学习率
