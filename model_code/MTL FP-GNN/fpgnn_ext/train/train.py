from argparse import Namespace
from logging import Logger
import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
from copy import deepcopy

from fpgnn_ext.tool.tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model
from fpgnn_ext.model import FPGNN
from fpgnn_ext.data import MoleDataSet

def epoch_train(model,data,loss_f,optimizer,scheduler,args):
    """定义训练模型一个 epoch 的函数，将一个epoch要分成多个batch训练"""
    model.train() #设置模型为训练模式，并对数据(MoleData)进行随机化处理
    data.random_data(args.seed)
    loss_sum = 0 #初始化损失值和已使用的数据量
    data_used = 0
    iter_step = args.batch_size #设置一个epoch中的batch大小,50
    
    for i in range(0,len(data),iter_step):
        if data_used + iter_step > len(data):
            break

        data_now = MoleDataSet(data[i:i+iter_step])
        smile = data_now.smile() #得到一个batch中所有的smile
        label = data_now.label() #得到一个batch中所有的标签

        #代码段的作用是根据标签数据创建掩码（mask）和目标（target），其中掩码用于指示标签是否为 None，目标将 None 替换为 0
        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])

        #为了确保 mask 和 target 与模型在同一个设备上
        if next(model.parameters()).is_cuda: #获取模型的第一个参数，并检查它是否在 GPU 上
            mask, target = mask.cuda(), target.cuda()

        #这段代码用于创建一个权重张量 weight，并根据是否需要在 GPU 上运行的条件将其移动到相应的设备上
        weight = torch.ones(target.shape)
        if args.cuda:
            weight = weight.cuda()

        #进行前向传播、计算损失、反向传播和参数更新
        model.zero_grad() #将梯度清零，确保每个训练迭代的梯度不会与上一次迭代的梯度累积，防止梯度爆炸或梯度消失问题
        pred = model(smile) #执行前向传播，获得预测输出 pred
        loss = loss_f(pred,target) * weight * mask #计算损失值，可以用于对不同样本或类别的重要性进行加权或忽略
        loss = loss.sum() / mask.sum() #为了计算平均每个样本的损失值，同时忽略那些被掩码标记为无效的样本
        loss_sum += loss.item() #记录整个训练过程中的累计损失
        data_used += len(smile) #更新已使用的数据量
        loss.backward() #执行反向传播，计算模型参数的梯度
        optimizer.step() #执行参数更新，使用优化器更新模型的参数值，使其朝着减小损失的方向前进
        if isinstance(scheduler, NoamLR): #isinstance() 函数用于检查一个对象是否是指定类型或类型元组中的一个实例
            scheduler.step() #这段代码用于在使用 Noam 学习率调度器时执行相应的学习率更新操作
    if isinstance(scheduler, ExponentialLR):
        scheduler.step() #这段代码用于在使用指数学习率调度器时执行相应的学习率更新操作

def predict(model,data,batch_size,scaler):
    """对给定的数据进行模型预测，在预测阶段，无需进行反向传播，不需要进行梯度计算和参数更新"""
    model.eval() # 将模型设置为评估模式
    pred = []
    data_total = len(data) #获取数据总数 data_total

    #将预测数据集分为多个batch进行预测
    for i in range(0,data_total,batch_size):
        data_now = MoleDataSet(data[i:i+batch_size])
        smile = data_now.smile()
        
        with torch.no_grad(): #使用 torch.no_grad() 上下文管理器来禁用梯度计算，以提高预测的效率
            pred_now = model(smile)
        
        pred_now = pred_now.data.cpu().numpy()
        
        if scaler is not None:
            ave = scaler[0]
            std = scaler[1]
            pred_now = np.array(pred_now).astype(float)
            change_1 = pred_now * std + ave
            pred_now = np.where(np.isnan(change_1),None,change_1)
        
        pred_now = pred_now.tolist()
        pred.extend(pred_now)
    
    return pred #用于对给定数据进行模型预测，并返回预测结果列表

def compute_score(pred,label,metric_f,args,log):
    """计算模型预测结果的评估分数"""
    info = log.info

    batch_size = args.batch_size
    task_num = args.task_num
    data_type = args.dataset_type
    
    if len(pred) == 0:
        return [float('nan')] * task_num
    
    pred_val = []
    label_val = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)
    
    result = []
    for i in range(task_num):
        if data_type == 'classification': #如果所有的标签值都是 0 或者都是 1，则输出警告信息并将评估分数设置为 NaN
            if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
                info('Warning: All labels are 1 or 0.')
                result.append(float('nan'))
                continue
            if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
                info('Warning: All predictions are 1 or 0.')
                result.append(float('nan'))
                continue
        re = metric_f(label_val[i],pred_val[i])
        result.append(re)
    
    return result #result 列表将包含每个任务的评估分数

def fold_train(args, log):
    """进行数据导入，数据集的划分，模型的训练和评估，将数据集按照8：1：1划分作用在于使用训练集建模，使用验证集调整超参数，最终使用测试集评估模型的性能"""
    info = log.info #记录调试信息
    debug = log.debug #输出调试信息
    
    debug('Start loading data') #输出一个调试信息，表示开始加载数据
    
    args.task_names = get_task_name(args.data_path) #获取任务名称,5
    data = load_data(args.data_path,args)  #一个MoleDataset对象，里面包含所有有效的MoleData
    args.task_num = data.task_num() #获取数据中的任务数量,5
    data_type = args.dataset_type #数据集的类型
    if args.task_num > 1: #断主要是用来判断是否进行多任务学习
        args.is_multitask = 1

    #这部分代码主要是对数据集进行划分和一些参数设置
    debug(f'Splitting dataset with Seed = {args.seed}.')
    if args.val_path:
        val_data = load_data(args.val_path,args) #加载验证集数据
    if args.test_path:
        test_data = load_data(args.test_path,args) #加载测试集数据
    if args.val_path and args.test_path: #已加载的数据集（data）赋值给训练集数据（train_data）(改)
        train_data = data
    elif args.val_path: #如果只有验证集路径存在，就将数据集划分成训练集和测试集即可
        split_ratio = (args.split_ratio[0],0,args.split_ratio[2])
        train_data, _, test_data = split_data(data,args.split_type,split_ratio,args.seed,log)
    elif args.test_path: ##如果只有测试集路径存在，就将数据集划分成训练集和验证集即可
       split_ratio = (args.split_ratio[0],args.split_ratio[1],0)
       train_data, val_data, _ = split_data(data,args.split_type,split_ratio,args.seed,log)
    else: #如果两个路径都没有，就按照原来的划分即可
        train_data, val_data, test_data = split_data(data,args.split_type,args.split_ratio,args.seed,log)
    debug(f'Dataset size: {len(data)}    Train size: {len(train_data)}    Val size: {len(val_data)}   Test size: {len(test_data)}')  ###改
    train_smi = pd.DataFrame(train_data.smile(),columns=['SMILES'])
    train_label = pd.DataFrame(train_data.label(),columns=['1A2','2C9','2C19','2D6','3A4'])
    pd.concat([train_smi,train_label],axis=1).to_csv("{}_train_datasets.csv".format(args.seed),index=False)

    val_smi = pd.DataFrame(val_data.smile(),columns=['SMILES'])
    val_label = pd.DataFrame(val_data.label(),columns=['1A2','2C9','2C19','2D6','3A4'])
    pd.concat([val_smi,val_label],axis=1).to_csv("{}_valid_datasets.csv".format(args.seed),index=False)

    test_smi = pd.DataFrame(test_data.smile(),columns=['SMILES'])
    test_label = pd.DataFrame(test_data.label(),columns=['1A2','2C9','2C19','2D6','3A4'])
    pd.concat([test_smi,test_label],axis=1).to_csv("{}_test_datasets.csv".format(args.seed),index=False)


    #打印出数据集的大小、训练集大小4828、验证集大小603和测试集大小605


    if data_type == 'regression':
        label_scaler = get_label_scaler(train_data) #如果是回归任务，则调用get_label_scaler()方法获取回归标签的缩放
    else:
        label_scaler = None
    args.train_data_size = len(train_data)
    
    loss_f = get_loss(data_type) #获取相应的损失函数
    metric_f = get_metric(args.metric) #获取相应的评估指标

    #模型的训练准备和初始化
    debug('Training Model') #输出一个调试信息，表示正在进行模型训练
    model = FPGNN(args)
    debug(model) #输出一个调试信息，打印出模型的详细信息
    if args.cuda:
        model = model.to(torch.device("cuda"))
    save_model(os.path.join(args.save_path, f'{args.seed}_model.pt'),model,label_scaler,args) #将模型及其参数保存到相应的路径中
    #os.path.join(目录名,文件名)函数将它们合并为一个完整的文件路径
    optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=0) #使用Adam优化器来优化模型的参数
    scheduler = NoamLR( optimizer = optimizer,  warmup_epochs = [args.warmup_epochs], total_epochs = None or [args.epochs] * args.num_lrs, \
                        steps_per_epoch = args.train_data_size // args.batch_size, init_lr = [args.init_lr], max_lr = [args.max_lr], \
                        final_lr = [args.final_lr] )
    #使用NoamLR调度器来调整学习率。在初始化时，设置了优化器、预热轮数、总共的训练轮数、每轮的步数、初始学习率、最大学习率和最终学习率
    #初始化最佳得分
    if data_type == 'classification':
        best_score = -float('inf') #其设置为负无穷
    else:
        best_score = float('inf')
    best_epoch = 0 #初始化最佳模型的训练轮数为0
    n_iter = 0 #初始化迭代次数为0
    
    epoch_train_list = []
    epoch_valid_list = []
    #是一个训练和评估模型的循环
    for epoch in range(args.epochs): #训练30个epoch，就是训练30次
        info(f'Epoch {epoch}') #输出当前轮数的信息
        
        epoch_train(model,train_data,loss_f,optimizer,scheduler,args) #代码调用epoch_train()函数来进行一轮的训练

        #接着，代码使用训练好的模型对训练集和验证集进行预测，并计算出相应的评估得分
        train_pred = predict(model,train_data,args.batch_size,label_scaler) #获得训练集的预测结果
        train_label = train_data.label() #获得训练集的真实标签
        train_score = compute_score(train_pred,train_label,metric_f,args,log) #模型对训练集的评估分数
        epoch_train_list.append(train_score)
        val_pred = predict(model,val_data,args.batch_size,label_scaler) ##获得验证集的预测结果
        val_label = val_data.label() ##获得验证集的真实标签
        val_score = compute_score(val_pred,val_label,metric_f,args,log) ##模型对验证集的评估分数
        epoch_valid_list.append(val_score)
        
        ave_train_score = np.nanmean(train_score) #计算出平均的训练集得分
        info(f'Train {args.metric} = {ave_train_score:.6f}')
        if args.task_num > 1:
            for one_name,one_score in zip(args.task_names,train_score):
                info(f'    train {one_name} {args.metric} = {one_score:.6f}')

        
        ave_val_score = np.nanmean(val_score) #计算出平均的验证集得分
        info(f'Validation {args.metric} = {ave_val_score:.6f}')
        if args.task_num > 1:
            for one_name,one_score in zip(args.task_names,val_score):
                info(f'    validation {one_name} {args.metric} = {one_score:.6f}')

        #更新最佳得分、最佳轮数和保存模型
        if data_type == 'classification' and ave_val_score > best_score:
            best_score = ave_val_score
            best_epoch = epoch
            save_model(os.path.join(args.save_path, f'{args.seed}_model.pt'),model,label_scaler,args)
        elif data_type == 'regression' and ave_val_score < best_score:
            best_score = ave_val_score
            best_epoch = epoch
            save_model(os.path.join(args.save_path, 'model.pt'),model,label_scaler,args)

    #输出最佳验证集得分和对应的轮数
    info(f'Best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

    #代码加载保存的最佳模型，并使用该模型对测试集进行预测，然后计算出测试集的评估得分(改)
    model = load_model(os.path.join(args.save_path, f'{args.seed}_model.pt'),args.cuda,log)
    #模型训练时，训练的是模型参数，得到最佳验证集得分，保存模型，保存的也是模型参数（权重和偏置），GAT、FPN、FFN
    test_smile = test_data.smile()
    test_label = test_data.label()
    
    test_pred = predict(model,test_data,args.batch_size,label_scaler)
    test_score = compute_score(test_pred,test_label,metric_f,args,log)
    
    ave_test_score = np.nanmean(test_score)
    #代码输出每个任务的测试集得分，并返回测试集得分作为结果
    info(f'Seed {args.seed} : test {args.metric} = {ave_test_score:.6f}')
    if args.task_num > 1:
        for one_name,one_score in zip(args.task_names,test_score):
            info(f'    test {one_name} {args.metric} = {one_score:.6f}')
    
    return epoch_train_list[best_epoch],epoch_valid_list[best_epoch],test_score
