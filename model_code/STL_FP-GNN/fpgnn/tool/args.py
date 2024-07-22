from argparse import ArgumentParser, Namespace
#ArgumentParser是用来定义和解析命令行参数
#Namespace是用来存储解析后的命令行参数
import torch
from .tool import mkdir

def add_train_argument(p):
    """向命令行解析器ArgumentParser对象（p）中添加一系列模型训练时所需的参数"""
    #使用.add_argument()方法向命令行解析器p中添加一个名为“--data_path”的参数
    #.add_argument('参数名称',type,help)
    #--data_path：是参数选项的名称，表示用户在命令行中需要指定这个参数的值
    #type:表示输入参数值的类型为字符串
    #hlep表示该参数的帮助信息，以便用户了解该参数的用途和使用方法
    #添加了一个输入数据的路径的参数
    p.add_argument('--data_path',type=str,default='random_10_seeds_3A4',
                   help='The path of input CSV file.')
    #添加一个输出数据路径的参数
    #default='model_save'：表示如果用户没有在命令行中指定 --save_path 参数的值，那么使用默认值 'model_save'
    p.add_argument('--save_path',type=str,default='STL_3A4_model_save',
                   help='The path to save output model.pt.,default is "model_save_val/"')
    #使用这个参数选项的目的是让用户在命令行中指定输出日志文件的保存路径
    #添加一个输出日志文件路径的参数
    p.add_argument('--log_path',type=str,default='Single_log',
                   help='The dir of output log file.')
    #使用这个参数选项的目的是让用户在命令行中指定数据集的类型
    #添加一个数据集类型的参数，且该参数值只能在choices中二选一，否咋就会报错
    #运行命令：python .py --dataset_type classification
    p.add_argument('--dataset_type',type=str,choices=['classification', 'regression'],default='classification',
                   help='The type of dataset.')
    #使用这个参数选项的目的是让用户在命令行中指定数据集是否为多任务数据集
    #添加一个是否是多任务数据集的参数，0表示不是多任务数据集，1表示是
    p.add_argument('--is_multitask',type=int,default=0,
                   help='Whether the dataset is multi-task. 0:no  1:yes.')
    #使用这个参数选项的目的是让用户在命令行中指定多任务训练中的任务数量
    #添加一个多任务数量的参数，默认值为1
    p.add_argument('--task_num',type=int,default=1,
                   help='The number of task in multi-task training.')
    #使用这个参数选项的目的是让用户在命令行中指定数据拆分（划分）的类型
    #添加一个数据集划分类型的参数，只能从'random'或者'scaffold'中选一，默认是'random'
    p.add_argument('--split_type',type=str,choices=['random', 'scaffold'],default='random',
                   help='The type of data splitting.')
    #使用这个参数选项的目的是让用户在命令行中指定数据集划分的比例
    #nargs=3：表示参数需要接受 3 个值。即用户在命令行中指定的值应该是 3 个浮点数，用空格或逗号分隔
    p.add_argument('--split_ratio',type=float,nargs=3,default=[0.8,0.1,0.1],
                   help='The ratio of data splitting.[train,valid,test]')
    #使用这个参数选项的目的是让用户在命令行中指定验证集的路径
    p.add_argument('--val_path',type=str,default=None,
                   help='The path of excess validation data.')
    #使用这个参数选项的目的是让用户在命令行中指定测试集的路径
    p.add_argument('--test_path',type=str,default=None,
                   help='The path of excess testing data.')
    #使用这个参数选项的目的是让用户在命令行中指定数据划分时的随机种子
    p.add_argument('--seed',type=int,default=0,
                   help='The random seed of model. Using in splitting data.')
    #使用这个参数选项的目的是让用户在命令行中指定交叉验证时划分的折数，默认为1
    p.add_argument('--num_folds',type=int,default=10,
                   help='The number of folds in cross validation.')
    #使用这个参数选项的目的是让用户在命令行中指定模型的评估指标
    #添加一个评估指标的参数，值只能在['auc','prc-auc','rmse']中三选一
    p.add_argument('--metric',type=str,choices=['auc', 'prc-auc', 'rmse'],default='auc',
                   help='The metric of data evaluation.')
    #使用这个参数选项的目的是让用户在命令行中指定模型训练的次数，默认训练30次
    p.add_argument('--epochs',type=int,default=30,
                   help='The number of epochs.')
    #使用这个参数选项的目的是让用户在命令行中设置批次大小，（即每个epoch中将数据集划分成多个batch训练），默认是50
    p.add_argument('--batch_size',type=int,default=50,
                   help='The size of batch.')
    #使用这个参数选项的目的是让用户在命令行中选择指纹类型，在['morgan','mixed']中二选一，默认使用'mixed'
    p.add_argument('--fp_type',type=str,choices=['morgan','mixed'],default='mixed',
                   help='The type of fingerprints. Use "morgan" or "mixed".')
    #使用这个参数选项的目的是让用户在命令行中指定隐藏层的维度大小，默认是300
    p.add_argument('--hidden_size',type=int,default=300,
                   help='The dim of hidden layers in model.')
    #使用这个参数选项的目的是让用户在命令行中指定 FPN 中第二个层的维度大小，默认是512
    p.add_argument('--fp_2_dim',type=int,default=600,
                   help='The dim of the second layer in fpn.')
    #使用这个参数选项的目的是让用户在命令行中指定 GNN 中注意力的维度大小
    p.add_argument('--nhid',type=int,default=40,
                   help='The dim of the attentions in gnn.')
    #使用这个参数选项的目的是让用户在命令行中指定 GNN 中的注意力头的数量
    p.add_argument('--nheads',type=int,default=5,
                   help='The number of the attentions in gnn.')
    #使用这个参数选项的目的是让用户在命令行中指定 GNN 在整个模型中所占的比例
    p.add_argument('--gat_scale',type=float,default=0.7,
                   help='The ratio of gnn in model.')
    #--dropout 参数允许用户在命令行中设置模型中的 Dropout 比例，以便在训练过程中对模型进行正则化，防止过拟合
    p.add_argument('--dropout',type=float,default=0.25,
                   help='The dropout of fpn and ffn.')
    #--dropout_gat 参数允许用户在命令行中设置图神经网络中的 Dropout 比例，以便在训练过程中对模型进行正则化，防止过拟合
    p.add_argument('--dropout_gat',type=float,default=0.45,
                   help='The dropout of gnn.')

def add_predict_argument(p):
    p.add_argument('--predict_path', type=str,default='random_10_seeds_2D6',
                   help='The path of input CSV file to predict.')
    p.add_argument('--result_path', type=str,default='case_study_results.txt',
                   help='The path of output CSV file.')
    p.add_argument('--model_path', type=str,default='STL_2D6_model_save',
                   help='The path of model.pt.')
    p.add_argument('--batch_size', type=int,default=1,
                   help='The size of batch.')

def add_hyper_argument(p):
    p.add_argument('--search_num', type=int,default=10,
                   help='The number of hyperparameters searching.')

def add_interfp_argument(p):
    p.add_argument('--log_path', type=str,default='log_fp',
                   help='The path of log file.')

def add_intergraph_argument(p):
    p.add_argument('--predict_path', type=str,default='final_external.csv',
                   help='The path of input CSV file to predict.')
    p.add_argument('--figure_path', type=str,default='figure',
                   help='The path of output figure file.')
    p.add_argument('--model_path', type=str,default='model_save_random/Seed_9/9_model.pt',
                   help='The path of model.pt.')
    p.add_argument('--batch_size', type=int,default=1,
                   help='The size of batch.')

def set_train_argument():
    p = ArgumentParser() #创建一个ArgumentParser对象，用于定义和解析命令行参数
    add_train_argument(p) #就可以向命令行解析器中添加一系列参数
    args = p.parse_args() #使用.parse_args()方法就可以解析命令行参数，返回Namespace对象

    #检查训练参数 args 中是否包含必要的参数 data_path 和 dataset_type，如果这两个参数不存在或为空，则会抛出一个 AssertionError 异常
    assert args.data_path
    assert args.dataset_type

    #mkdir(args.save_path) 这行代码用于创建一个目录，目录的名称由参数 args.save_path 指定
    mkdir(args.save_path)
    
    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'

    #用于检查命令行参数的合法性
    if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
        raise ValueError('Metric or data_type is error.')
    if args.dataset_type == 'regression' and args.metric not in ['rmse']:
        raise ValueError('Metric or data_type is error.')
    if args.fp_type not in ['mixed','morgan']:
        raise ValueError('Fingerprint type is error.')

    #为训练参数args添加一些默认值
    #args.cuda：该参数表示是否使用GPU加速，它的值被设定为 torch.cuda.is_available()，即如果有可用的GPU，则值为 True，否则为 False
    args.cuda = torch.cuda.is_available()
    args.init_lr = 1e-4 #初始学习率，被设定为 1e-4，即 0.0001
    args.max_lr = 1e-3 #最大学习率，被设定为 1e-3，即 0.001
    args.final_lr = 1e-4 #表示最终学习率，被设定为 1e-4，即 0.0001
    args.warmup_epochs = 2.0 #表示学习率预热的轮数（epoch），被设定为 2.0
    args.num_lrs = 1 #表示学习率变化的阶段数，被设定为 1
    
    return args

def set_predict_argument():
    p = ArgumentParser()
    add_predict_argument(p)
    args = p.parse_args()
    
    assert args.predict_path
    assert args.model_path
    
    args.cuda = torch.cuda.is_available()
    
    mkdir(args.result_path, isdir = False)
    
    return args

def set_hyper_argument():
    p = ArgumentParser()
    add_train_argument(p)
    add_hyper_argument(p)
    args = p.parse_args()
    
    assert args.data_path
    assert args.dataset_type
    
    mkdir(args.save_path)
    
    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'

    if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
        raise ValueError('Metric or data_type is error.')
    if args.dataset_type == 'regression' and args.metric not in ['rmse']:
        raise ValueError('Metric or data_type is error.')
    if args.fp_type not in ['mixed','morgan']:
        raise ValueError('Fingerprint type is error.')

    args.cuda = torch.cuda.is_available()
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs = 2.0
    args.num_lrs = 1
    args.search_now = 0
    
    return args

def set_interfp_argument():
    p = ArgumentParser()
    add_predict_argument(p)
    add_interfp_argument(p)
    args = p.parse_args()
    
    assert args.predict_path
    assert args.model_path
    
    args.cuda = torch.cuda.is_available()
    args.fp_changebit = 0
    
    mkdir(args.result_path, isdir = False)
    
    return args

def set_intergraph_argument():
    p = ArgumentParser()
    add_intergraph_argument(p)
    args = p.parse_args()
    
    
    assert args.predict_path
    assert args.model_path
    
    args.cuda = torch.cuda.is_available()
    args.inter_graph = 1
    
    mkdir(args.figure_path, isdir = True)
    
    return args