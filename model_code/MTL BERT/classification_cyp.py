#import tensorflow as tf
#import tensorflow.keras as keras
#import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import Prediction_Dataset, Pretrain_Collater, Finetune_Collater,randomize_smile
#from sklearn.metrics import r2_score,roc_auc_score
from metrics import AverageMeter,Metric

import os
from model import PredictionModel,BertModel
import argparse

#设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #设置了 TensorFlow 使用 GPU 时，允许显存的增长
#TensorFlow 将根据需要动态分配显存，以更有效地利用 GPU 资源
#keras.backend.clear_session() #清除了 Keras 的会话，释放之前分配的资源
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" #使用该选项来选择要使用的 GPU 设备。这里将其设置为 "0"，表示使用第一个可用的 GPU 设备


parser = argparse.ArgumentParser() #创建了一个命令行参数解析器
#配置一些参数
parser.add_argument('--smiles-head', nargs='+', default=['SMILES'], type=str)
parser.add_argument('--clf-heads', nargs='+', default=['1A2','2C9','2C19','2D6','3A4'], type=str)
parser.add_argument('--aug_num', type=int, default=10, choices=[1,5,10,20,50,100],help='No. of output perceptron (class)')
parser.add_argument('--result_path',type=str,default='Bert_model_seed/',help='The path of model saved.')
args = parser.parse_args() #解析参数，返回Namespace对象（字典）

#'Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub'
# 'caco2', 'logD','logS','tox','PPB'

def main(seed):
    # tasks = ['Ames', 'BBB', 'FDAMDD', 'H_HT', 'Pgp_inh', 'Pgp_sub']
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # tasks = ['BBB', 'FDAMDD',  'Pgp_sub']

    #不同规模的BERT模型结构，如小型、中型、大型网络等，包括层数、头数、隐层维度等参数
    small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights'}
    medium = {'name':'Medium','num_layers': 8, 'num_heads': 8, 'd_model': 256,'path':'medium_weights'}
    large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 512,'path':'large_weights'}

    arch = medium  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 10

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']

    dff = d_model * 4
    vocab_size = 60
    dropout_rate = 0.1

    #设置随机种子，用于使实验可重复
    seed = seed
    np.random.seed(seed=seed) #设置 NumPy 随机数生成器的随机种子
    torch.cuda.manual_seed(seed) #设置 TensorFlow 随机数生成器的随机种子

    #从数据文件中读取数据集，创建多标签数据集，形式：左侧SMILES，右侧多个标签
    # dfs = []
    # columns = set() #确保后续的数据合并不会出现重复的列
    # for reg_head in args.reg_heads:
    #     df = pd.read_csv('data/reg/{}.csv'.format(reg_head))
    #     df[reg_head] = (df[reg_head]-df[reg_head].mean())/(df[reg_head].std())
    #     dfs.append(df)
    #     columns.update(df.columns.to_list())
    # for clf_head in args.clf_heads:
    #     df = pd.read_csv('{}.csv'.format(clf_head))
    #     dfs.append(df)
    #     columns.update(df.columns.to_list()) # df 数据框的列名添加到 columns 集合中
    #
    # train_temps = []
    # test_temps = []
    # valid_temps = []
    # columns = list(columns)
    #
    # for df in dfs:
    #     temp = pd.DataFrame(index=range(len(df)),columns=columns) #初始化一个空白的数据框
    #     for column in df.columns:
    #         temp[column] = df[column]#相当于复制df
    #     temp = temp.sample(frac=1).reset_index(drop=True) #它将数据随机重排（相当于打乱了数据的顺序），然后重新设置了索引
    #     train_temp = temp[:int(0.8*len(temp))] #80% 部分划分为训练集
    #     train_temps.append(train_temp)
    #
    #     test_temp = temp[int(0.8*len(temp)):int(0.9*len(temp))] #中间 10% 划分为测试集
    #     test_temps.append(test_temp)
    #
    #     valid_temp = temp[int(0.9*len(temp)):] #最后10% 划分为验证集
    #     valid_temps.append(valid_temp)
    #
    # train_df = pd.concat(train_temps,axis=0).reset_index(drop=True) #训练集
    # test_df = pd.concat(test_temps, axis=0).reset_index(drop=True) #测试集
    # valid_df = pd.concat(valid_temps, axis=0).reset_index(drop=True) #验证集

    ####使用已经划分好的数据集进行训练，测试
    train_df = pd.read_csv(r'{}_train_datasets.csv'.format(seed))
    valid_df = pd.read_csv(r'{}_valid_datasets.csv'.format(seed))
    test_df = pd.read_csv(r'{}_test_datasets.csv'.format(seed))

    ####额外添加，数据增强部分：
    if args.aug_num > 1:
        train_temp = pd.concat([train_df] * (args.aug_num - 1), axis=0)
        train_temp["SMILES"] = train_temp["SMILES"].map(lambda x: randomize_smile(x))
        train_df = pd.concat([train_temp, train_df], ignore_index=True)  ####数据增强后的SMILES序列
        #train_smi = train_df.iloc[:,0]
        #train_label = pd.concat([train_df.iloc[:,1],train_df.iloc[:,3],train_df.iloc[:,5],train_df.iloc[:,7],train_df.iloc[:,9],
                                 #train_df.iloc[:,11],train_df.iloc[:,13],train_df.iloc[:,15],train_df.iloc[:,17],train_df.iloc[:,19]],axis=1)
        #train_total = pd.concat([train_smi,train_label],axis=1)
    if args.aug_num > 1:
        val_temp = pd.concat([valid_df] * (args.aug_num - 1), axis=0)
        val_temp["SMILES"] = val_temp["SMILES"].map(lambda x: randomize_smile(x))
        valid_df = pd.concat([val_temp, valid_df], ignore_index=True)
        #valid_smi = valid_df.iloc[:,0]
        #valid_label = pd.concat([valid_df.iloc[:, 1], valid_df.iloc[:, 3], valid_df.iloc[:, 5], valid_df.iloc[:, 7], valid_df.iloc[:, 9],
             #valid_df.iloc[:, 11], valid_df.iloc[:, 13], valid_df.iloc[:, 15], valid_df.iloc[:, 17], valid_df.iloc[:, 19]], axis=1)
        #valid_total = pd.concat([valid_smi, valid_label], axis=1)
    if args.aug_num > 1:
        test_temp = pd.concat([test_df] * (args.aug_num - 1),axis=0)
        test_temp['SMILES'] = test_temp['SMILES'].map(lambda x: randomize_smile(x))
        test_df = pd.concat([test_temp,test_df],ignore_index=True)
        #test_smi = test_df.iloc[:, 0]
        #test_label = pd.concat(
            #[test_df.iloc[:, 1], test_df.iloc[:, 3], test_df.iloc[:, 5], test_df.iloc[:, 7], test_df.iloc[:, 9],
             #test_df.iloc[:, 11], test_df.iloc[:, 13], test_df.iloc[:, 15], test_df.iloc[:, 17],
             #test_df.iloc[:, 19]], axis=1)
        #test_total = pd.concat([test_smi, test_label], axis=1)

    #将数据集转化成带有任务token的数值化的SMIELS
    train_dataset = Prediction_Dataset(train_df, smiles_head=args.smiles_head,
                                                               clf_heads=args.clf_heads)
    test_dataset = Prediction_Dataset(test_df, smiles_head=args.smiles_head,
                                       clf_heads=args.clf_heads)
    valid_dataset = Prediction_Dataset(valid_df, smiles_head=args.smiles_head,
                                       clf_heads=args.clf_heads)

    #创建数据加载器（DataLoader）对象，训练数据的随机洗牌以增加训练的随机性，将数值化的SMIELS和标签处理成多个batch
    train_dataloader = DataLoader(train_dataset, batch_size=64,shuffle=True,collate_fn=Finetune_Collater(args))
    test_dataloader = DataLoader(test_dataset, batch_size=128,shuffle=False,collate_fn=Finetune_Collater(args))
    valid_dataloader = DataLoader(valid_dataset, batch_size=128,shuffle=False,collate_fn=Finetune_Collater(args))

    #使用预训练的参数（测试集acc最高）初始化模型的权重和偏置，微调部分的预训练（编码层参数和BERT模型的预训练编码层参数一样）
    # x, property = next(iter(train_dataset))
    model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dropout_rate=0.1,clf_nums=len(args.clf_heads))
    model.encoder.load_state_dict(torch.load('weights/medium_weights_bert_encoder_weightsmedium_50.pt')) #导入EncoderLayer层的状态字典（权重和偏置）
    ###导入BERT模型编码层的参数初始化微调模型预训练编码层的参数
    model = model.to(device)
    # if pretraining:
    #     model.encoder.load_state_dict(torch.load())
    #     print('load_wieghts')

    #创建了一个 AdamW 优化器，用于优化模型的参数
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.5e-4,betas=(0.9,0.98))
    # lm = lambda x:x/10*(5e-5) if x<10 else (5e-5)*10/x
    # lms = LambdaLR(optimizer,[lm])

    #这些变量用于跟踪训练、测试和验证阶段的损失和acc
    train_loss = AverageMeter()
    test_loss = AverageMeter()
    valid_loss = AverageMeter()

    #用于跟踪训练、测试和验证阶段的 AUC（Area Under the Curve）值
    train_aucs = Metric()
    test_aucs = Metric()
    valid_aucs = Metric()

    #用于跟踪训练、测试和验证阶段的 R-squared 值
    # train_r2 = Records_R2()
    # test_r2 = Records_R2()
    # valid_r2 = Records_R2()

    #定义了两个损失函数。loss_func1 是二分类交叉熵损失函数（BCEWithLogitsLoss），用于分类任务。loss_func2 是均方误差损失函数（MSELoss），用于回归任务
    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')
    #loss_func2 = torch.nn.MSELoss(reduction='none')

    # 这个变量可能用于监视训练的停止条件
    stopping_monitor = 0

    #定义一个训练步骤
    def train_step(x,properties):
        model.train() #将模型设置为训练模式
        #从输入的 properties 字典中获取分类任务（clf_true）和回归任务（reg_true）的真实标签
        clf_true = properties['clf']
        #reg_true = properties['reg']

        #得到分类和回归的预测标签
        properties_pred = model(x) #进行前向传播
        clf_pred = properties_pred['clf']
        #reg_pred = properties_pred['reg']

        #初始化损失值为零
        loss = 0

        #loss_func1 计算分类任务的损失，这是二分类交叉熵损失函数,将损失限制在非缺失值位置上。
        #将计算的损失累积到总损失 loss 中
        if len(args.clf_heads)>0:
            loss += (loss_func1(clf_pred,clf_true*(clf_true!=-1000).float())*(clf_true!=-1000).float()).sum()/((clf_true!=-1000).float().sum()+1e-6)

        #使用 loss_func2 计算回归任务的损失，这是均方误差损失函数。
        #将损失限制在非缺失值位置上（reg_true!=-1000）。
        #if len(args.reg_heads) > 0:
            #loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / ((reg_true != -1000).float().sum()+1e-6)

        #进行梯度清零，进行反向传播计算梯度，进行参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #在训练过程中更新性能指标和损失
        if len(args.clf_heads) > 0:
            train_aucs.update(clf_pred.detach().cpu(),clf_true.detach().cpu()) #更新分类任务的性能指标（AUC）
        #if len(args.reg_heads) > 0:
            #train_r2.update(reg_pred.detach().cpu(),reg_true.detach().cpu()) #更新回归任务的性能指标（R^2 分数）
        train_loss.update(loss.detach().cpu().item(),x.shape[0]) #更新训练损失值和平均损失


    #定义一个测试步骤
    def test_step(x, properties):
        model.eval() #将模型设置为评估模式
        with torch.no_grad(): #禁用梯度计算，加快计算速度

            #真实的分类和回归标签
            clf_true = properties['clf']
            #reg_true = properties['reg']
            #预测的分类和回归标签
            properties_pred = model(x)
            clf_pred = properties_pred['clf']
            #reg_pred = properties_pred['reg']

            #初始化损失值
            loss = 0

            #在测试过程中更新性能指标和损失
            if len(args.clf_heads) > 0:
                loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                            clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum()+1e-6)

            #if len(args.reg_heads) > 0:
                #loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / ((reg_true != -1000).sum()+1e-6)

            if len(args.clf_heads) > 0:
                test_aucs.update(clf_pred.detach().cpu(), clf_true.detach().cpu())
            #if len(args.reg_heads) > 0:
                #test_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())


    #定义一个验证的步骤
    def valid_step(x, properties):
        model.eval() #将模型设为验证模式
        with torch.no_grad(): #禁用梯度计算
            clf_true = properties['clf']
            #reg_true = properties['reg']

            properties_pred = model(x)

            clf_pred = properties_pred['clf']
            #reg_pred = properties_pred['reg']

            loss = 0

            if len(args.clf_heads) > 0:
                loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                            clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum()+1e-6)

            #if len(args.reg_heads) > 0:
                #loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / ((reg_true != -1000).sum()+1e-6)

            if len(args.clf_heads) > 0:
                valid_aucs.update(clf_pred.detach().cpu(), clf_true.detach().cpu())
            #if len(args.reg_heads) > 0:
                #valid_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
            valid_loss.update(loss.detach().cpu().item(),x.shape[0]) ###更新验证损失值和平均损失


    min_val_auc = 0
    stopping_monitor = 0
    train_auc_list = []
    valid_auc_list = []
    #开始进行训练，测试和验证
    for epoch in range(100): ####一共进行100次训练，为了避免过拟合也可采用Early Stopping
        #进行训练（微调）
        for x,properties in train_dataloader: #将训练集按照batch进行训练
            train_step(x,properties)

        print('epoch: ',epoch,'train loss: {:.4f}'.format(train_loss.avg)) #输出一个epoch的平均损失
        if len(args.clf_heads)>0:
            clf_results = train_aucs.roc_auc_score() #得到五个分类任务的AUC值
            train_auc_list.append(clf_results)
            #train_auc_mean = np.mean(clf_results)
            for num,clf_head in enumerate(args.clf_heads):
                print('train auc {}: {:.4f}'.format(clf_head,clf_results[num])) #输出一个epoch每个任务的AUC值
        #if len(args.reg_heads) > 0:
            #reg_results = train_r2.results()
            #for num,reg_head in enumerate(args.reg_heads):
                #print('train r2 {}: {:.4f}'.format(reg_head,reg_results[num]))

        #重置一个epoch的AUC，r2，损失值
        train_aucs.reset()
        #train_r2.reset()
        train_loss.reset()

        # 进行验证的过程
        #进行验证的过程
        for x, properties in valid_dataloader: #将验证集按照batch进行验证
            valid_step(x, properties)
        print('epoch: ',epoch,'valid loss: {:.4f}'.format(valid_loss.avg)) #获得一个epoch的损失均值
        if len(args.clf_heads) > 0:
            clf_results = valid_aucs.roc_auc_score()
            valid_auc_mean = np.mean(clf_results)
            valid_auc_list.append(clf_results)
            for num, clf_head in enumerate(args.clf_heads):
                print('valid auc {}: {:.4f}'.format(clf_head, clf_results[num])) #输出一个epoch每个任务的AUC值
        #if len(args.reg_heads) > 0:
            #reg_results = valid_r2.results()
            #for num, reg_head in enumerate(args.reg_heads):
                #print('valid r2 {}: {:.4f}'.format(reg_head, reg_results[num]))

        ###早停策略
        if valid_auc_mean > min_val_auc:
            min_val_auc = valid_auc_mean
            stopping_monitor = 0
            best_epoch = epoch
            print('Best epoch: {}'.format(epoch))
            print('Best val auc: {:.4f}'.format(min_val_auc))
            torch.save(model.state_dict(),os.path.join(args.result_path,'BERT_{}_model_aug_10_weights.pt'.format(seed)))
        else:
            stopping_monitor += 1
        if stopping_monitor > 0:
            print('Stopping_monitor: {} ', stopping_monitor)
        if stopping_monitor > 20:
            break

        # 重置一个epoch的AUC，r2，损失值
        valid_aucs.reset()
        #valid_r2.reset()
        valid_loss.reset()

    #进行测试的过程
    model.load_state_dict(torch.load(os.path.join(args.result_path,'BERT_{}_model_aug_10_weights.pt'.format(seed))))

    for x, properties in test_dataloader: #将测试集按照batch进行测试
        test_step(x, properties)
    print('test loss: {:.4f}'.format(test_loss.avg)) #获得一个epoch的损失均值
    if len(args.clf_heads) > 0:
        test_results = np.array(test_aucs.roc_auc_score())
        test_acc = np.array(test_aucs.accuracy_score())
        test_bacc = np.array(test_aucs.bacc())
        test_sp = np.array(test_aucs.specifity())
        test_re = np.array(test_aucs.recall())
        test_f1 = np.array(test_aucs.f1())
        test_pre = np.array(test_aucs.pre())
        test_mcc = np.array(test_aucs.mcc())

        for num, clf_head in enumerate(args.clf_heads):
            print('test auc {}: {:.4f}'.format(clf_head, test_results[num])) ##输出一个epoch每个任务的AUC值
    best_train = np.array(train_auc_list[best_epoch])
    best_val = np.array(valid_auc_list[best_epoch])

    return best_train,best_val,test_results,test_acc,test_bacc,test_sp,test_re,test_f1,test_pre,test_mcc

if __name__ == '__main__':
    seeds = [0,1,2,3,4,5,6,7,8,9]
    best_val_list = []
    best_train_list = []
    test_auc_list = []
    test_acc_list = []
    test_recall_list = []
    test_precision_list = []
    test_f1_list = []
    test_bacc_list = []
    test_sp_list = []
    test_mcc_list = []
    for seed in seeds:
        best_train,best_val,test_results,test_acc,test_bacc,test_sp,test_re,test_f1,test_pre,test_mcc = main(seed)
        best_train_list.append(best_train)
        best_val_list.append(best_val)
        test_auc_list.append(test_results)
        test_acc_list.append(test_acc)
        test_bacc_list.append(test_bacc)
        test_sp_list.append(test_sp)
        test_recall_list.append(test_re)
        test_f1_list.append(test_f1)
        test_precision_list.append(test_pre)
        test_mcc_list.append(test_mcc)

    train_df = pd.DataFrame(best_train_list, columns=['train_1A2', 'train_2C9', 'train_2C19', 'train_2D6', 'train_3A4'])
    val_df = pd.DataFrame(best_val_list, columns=['valid_1A2', 'valid_2C9', 'valid_2C19', 'valid_2D6', 'valid_3A4'])
    auc_df = pd.DataFrame(test_auc_list, columns=['test_1A2', 'test_2C9', 'test_2C19', 'test_2D6', 'test_3A4'])
    total_auc = pd.concat([train_df, val_df, auc_df], axis=1)
    total_auc.to_csv('Bert_model_seed/10_seeds_AUC_BERT.csv', index=False)
    acc_df = pd.DataFrame(test_acc_list, columns=args.clf_heads)
    acc_df.to_csv('Bert_model_seed/10_seeds_ACC_BERT.csv', index=False)
    recall_df = pd.DataFrame(test_recall_list, columns=args.clf_heads)
    recall_df.to_csv('Bert_model_seed/10_seeds_Recall_BERT.csv', index=False)
    pre_df = pd.DataFrame(test_precision_list, columns=args.clf_heads)
    pre_df.to_csv('Bert_model_seed/10_seeds_precision_BERT.csv', index=False)
    f1_df = pd.DataFrame(test_f1_list, columns=args.clf_heads)
    f1_df.to_csv('Bert_model_seed/10_seeds_f1_score_BERT.csv', index=False)
    bacc_df = pd.DataFrame(test_bacc_list, columns=args.clf_heads)
    bacc_df.to_csv('Bert_model_seed/10_seeds_BACC_BERT.csv', index=False)
    sp_df = pd.DataFrame(test_sp_list, columns=args.clf_heads)
    sp_df.to_csv('Bert_model_seed/10_seeds_Specificity_BERT.csv', index=False)
    mcc_df = pd.DataFrame(test_mcc_list, columns=args.clf_heads)
    mcc_df.to_csv('Bert_model_seed/10_seeds_MCC_BERT.csv', index=False)










