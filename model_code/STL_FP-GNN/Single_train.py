from argparse import Namespace
from logging import Logger
import numpy as np
import os
import random
import torch
from fpgnn.train import fold_train
from fpgnn.tool import set_log, set_train_argument, get_task_name, mkdir
import pandas as pd

def set_random_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)   
    torch.manual_seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)  ##设置 Python 散列的种子
    if args.cuda:  ##设置了 CUDA 的随机数生成器的种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def training(args,log):
    """是一个模型训练和评估的函数"""
    info = log.info

    #data_path = args.data_path
    train_scores = []
    valid_scores = []
    test_scores = []
    #使用不同的种子进行多次交叉验证
    for num_fold in range(args.num_folds):
        args.seed = num_fold
        set_random_seed(args)
        mkdir(args.save_path)
        info(f'Seed {args.seed}')  # 在每个交叉验证的开始，输出当前的随机种子
        train_score,valid_score,test_score = fold_train(args,log) #进行一次交叉验证（一个种子/一次划分）训练好模型的结果，该函数接受参数和日志对象，并返回一个得分（改）

        train_scores.append(train_score)
        valid_scores.append(valid_score)
        test_scores.append(test_score)
    #train_df = pd.DataFrame(train_scores,columns=['train_1A2','train_2C9','train_2C19','train_2D6','train_3A4'])
    train_df = pd.DataFrame(train_scores, columns=[f'train_{args.task}'])
    valid_df = pd.DataFrame(valid_scores,columns=[f'valid_{args.task}'])
    test_df = pd.DataFrame(test_scores,columns=[f'test_{args.task}'])
    total_df = pd.concat([train_df,valid_df,test_df],axis=1)
    total_df.to_csv(f'random_Single_{args.task}_FP_GNN_10_seeds_AUC_result.csv')
    score = np.array(test_scores)
    
    info(f'Running {args.num_folds} folds in total.') #输出总共进行了多少次交叉验证
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(test_scores): #代码依次输出每次交叉验证的种子，并计算平均得分
            info(f'Seed { num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')
            if args.task_num > 1: #代码输出每个任务的得分
                for one_name,one_score in zip(args.task_names,fold_score):
                    info(f'Task {one_name} {args.metric} = {one_score:.6f}')
    ave_task_score = np.nanmean(score, axis=1) #每个任务的平均得分
    score_ave = np.nanmean(ave_task_score) #所有任务的平均得分以及其标准差
    score_std = np.nanstd(ave_task_score)
    info(f'Average test {args.metric} = {score_ave:.6f} +/- {score_std:.6f}')

    #args.task_names = get_task_name(args.data_path)
    #if args.task_num > 1:
        #for i,one_name in enumerate(args.task_names):
            #info(f'average all-fold {one_name} {args.metric} = {np.nanmean(score[:, i]):.6f} +/- {np.nanstd(score[:, i]):.6f}')
    
    
if __name__ == '__main__':
    args = set_train_argument() #先设置训练参数和日志对象
    log = set_log('train',args.log_path)
    tasks = ['1A2','2C9','2C19','2D6','3A4']
    for task in tasks:
        args.task = task
        print(f"{args.task} Training")
        args.save_path = f'random_STL_{args.task}_model_save'
        args.data_path = f'random_10_seeds_{args.task}'
        training(args,log) #进行模型训练和评估