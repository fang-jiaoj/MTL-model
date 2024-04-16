import pandas as pd
import numpy as np
import scipy
import random
import os
from argparse import ArgumentParser
from Attentive_utils.configure_attentivefp import attentivefp_configure

import dgl
import torch
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping
from Attentive_utils.eval_meter import Meter
from Attentive_utils.featurizers import CanonicalAtomFeaturizer
from Attentive_utils.featurizers import CanonicalBondFeaturizer
from dgllife.utils import one_hot_encoding
from Attentive_utils.attentivefp_predictor_IG import AttentiveFPPredictor
from dgllife.data.csv_dataset import MoleculeCSVDataset


###设置设备
if torch.cuda.is_available():
    print("Use GPU")
    device = 'cuda'
else:
    print("Use CPU")
    device = 'cpu'

#设置随机数种子
###设置随机数种子
def set_random_seed(args):
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  ##设置 Python 散列的种子
    if torch.cuda.is_available():  ##设置了 CUDA 的随机数生成器的种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

###定义一个批次处理函数
def collate_molgraphs(data):
    assert len(data[0]) in [3,4],\
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles,graphs,labels = map(list,zip(*data))
        masks = None
    else:
        smiles,graphs,labels,masks = map(list,zip(*data))

    bg = dgl.batch(graphs) ##把一个batch的图组成一张大图
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    labels = torch.stack(labels,dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks,dim=0)
    return smiles,bg,labels,masks

###将数据转变为图数据
def load_data(args,data):
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he',self_loop=args['self_loop'])
    n_feats = atom_featurizer.feat_size('hv')
    e_feats = bond_featurizer.feat_size('he')
    dataset = MoleculeCSVDataset(data,smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=args['self_loop']),
                                 node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=args['data_path'] + '_graph.bin',
                                 task_names=args['task_names'],
                                 load=False, init_mask=True, n_jobs=1)
    return dataset,n_feats,e_feats

def run_a_train_epoch(args,epoch,model,data_loader,loss_criterion,optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id,batch_data in enumerate(data_loader):
        smiles,bg,labels,masks = batch_data
        bg = bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg,n_feats,e_feats)
        loss = (loss_criterion(prediction,labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction,labels,masks)
        losses.append(loss.data.item())
    train_loss_avg = np.mean(losses)  ###得到一个epoch的损失
    train_score = train_meter.compute_metric(args['metric'])
    train_score_avg = np.mean(train_score)
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score_avg))  # 打印当前 epoch 的训练指标得分
    if args['n_tasks'] > 1:
        for one_name, one_score in zip(args['task_names'], train_score):
            print('train {} {} = {:.4f}'.format(one_name, args['metric'], one_score))
    return train_score

###定义一个epoch的验证函数
def run_a_val_epoch(args,model,data_loader,loss_criterion):
    model.eval()
    val_losses = []
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id,batch_data in enumerate(data_loader):
            smiles,bg,labels,masks = batch_data
            bg,labels = bg.to(device),labels.to(device)
            masks = masks.to(device)
            n_data = bg.ndata.pop('hv').to(device)
            e_data = bg.edata.pop('he').to(device)
            val_prediction = model(bg,n_data,e_data)
            loss = (loss_criterion(val_prediction,labels) * (masks != 0).float()).mean()
            val_loss = loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(val_prediction, labels, masks)
        val_loss_avg = np.mean(val_losses)  ###得到一个epoch的损失
        val_score = eval_meter.compute_metric(args['metric'])
        return val_score

###定义一个测试函数
def run_a_test(args,model,data_loader,loss_criterion):
    model.eval()
    test_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            test_prediction = model(bg, n_feats,e_feats)
            test_meter.update(test_prediction, labels, masks)
        test_auc = test_meter.compute_metric(args['metric'])
        test_acc = test_meter.compute_metric('accuracy')
        test_recall = test_meter.compute_metric('recall')
        test_precision = test_meter.compute_metric('precision')
        test_f1 = test_meter.compute_metric('f1')
        test_bacc = test_meter.compute_metric('bacc')
        test_sp = test_meter.compute_metric('sp')
        test_mcc = test_meter.compute_metric('mcc')
        return test_auc,test_acc,test_recall,test_precision,test_f1,test_bacc,test_sp,test_mcc

def train(args):
    train_data = pd.read_csv("{}_train_datasets.csv".format(args['seed']))
    valid_data = pd.read_csv("{}_valid_datasets.csv".format(args['seed']))
    test_data = pd.read_csv("{}_test_datasets.csv".format(args['seed']))

    ###把数据处理成图
    train_set,n_feats,e_feats = load_data(args,train_data)
    valid_set,n_feats,e_feats = load_data(args,valid_data)
    test_set,n_feats,e_feats = load_data(args,test_data)

    ###把数据划分成多个batch
    train_dataloader = DataLoader(train_set,batch_size=args['batch_size'],shuffle=True,collate_fn=collate_molgraphs)
    valid_dataloader = DataLoader(valid_set,batch_size=args['batch_size'],collate_fn=collate_molgraphs)
    test_dataloader = DataLoader(test_set,batch_size=args['batch_size'],collate_fn=collate_molgraphs)

    ###定义模型
    model = AttentiveFPPredictor(node_feat_size=n_feats,edge_feat_size=e_feats,num_layers=args['num_layers'],
                                 num_timesteps=args['num_timesteps'],graph_feat_size=args['graph_feat_size'],
                                 predictor_hidden_feats=args['predictor_hidden_feats'],n_tasks=5,dropout=args['dropout'])
    model = model.to(device)

    ###定义损失函数
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    ##定义一个Adam优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
    ###定义一个早停的类
    stopper = EarlyStopping(mode='higher',patience=args['patience'])

    train_score_list = []
    val_score_list = []
    val_score_avg_list = []
    for epoch in range(args['num_epochs']):
        train_score = run_a_train_epoch(args, epoch, model, train_dataloader, loss_fn, optimizer)
        val_score = run_a_val_epoch(args, model, valid_dataloader, loss_fn)
        val_score_avg = np.mean(val_score)
        train_score_list.append(train_score)
        val_score_list.append(val_score)
        val_score_avg_list.append(val_score_avg)
        early_stop = stopper.step(val_score_avg, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score_avg, args['metric'], stopper.best_score))  # stopper.best_score记录整个训练过程中验证集上最好的性能指标分数
        if args['n_tasks'] > 1:
            for one_name, one_score in zip(args['task_names'], val_score):
                print('Validation {} {} = {:.4f}'.format(one_name, args['metric'], one_score))

        if early_stop:
            break

    ###加载验证集上最佳的模型
    stopper.load_checkpoint(model)
    torch.save(model.state_dict(), os.path.join(args['result_path'], 'MultiAttentiveFP_{}_model.pt'.format(args['seed'])))

    ###进行测试
    test_auc, test_acc, test_recall, test_precision, test_f1, test_bacc, test_sp, test_mcc = \
        run_a_test(args, model, test_dataloader, loss_fn)
    test_score_avg = np.mean(test_auc)
    print('Test {} {:.4f}'.format(args['metric'], test_score_avg))  # 计算测试集的性能指标
    if args['n_tasks'] > 1:
        for one_name, one_score in zip(args['task_names'], test_auc):
            print('Test {} {} = {:.4f}'.format(one_name, args['metric'], one_score))

    ###输出AUC的结果
    max_index = np.argmax(val_score_avg_list)
    best_val = val_score_list[max_index]
    best_train = train_score_list[max_index]
    return best_val, best_train, test_auc, test_acc, test_recall, test_precision, test_f1, test_bacc, test_sp, test_mcc

if __name__ == '__main__':
    parser = ArgumentParser()  ###设置一个参数解析器
    parser.add_argument('--data_path',type=str,default='Multi_AttentiveFP_data/')
    parser.add_argument('--result_path',type=str,default='Multi_AttentiveFP_model/')
    parser.add_argument('--task_names',type=str,default='1A2,2C9,2C19,2D6,3A4',
                        help='The name of tasks.')
    parser.add_argument('--seed',type=int,default=0,help="The seed of random.")
    parser.add_argument('--n_tasks',type=int,default=5,help='The number of tasks.')
    parser.add_argument('--num_folds',type=int,default=10,help='The number of folds in cross validation.')

    args = parser.parse_args().__dict__  ##参数被解析为字典的方式存储在args
    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',')
        args.update(attentivefp_configure(args))

    seed_first = args['seed']
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
    for num_fold in range(args['num_folds']):
        args['seed'] = seed_first + num_fold
        print('Seed {}'.format(args['seed']))
        best_val, best_train, test_auc, test_acc, test_recall, test_precision, test_f1, test_bacc, test_sp, test_mcc = train(
            args)
        best_val_list.append(best_val)
        best_train_list.append(best_train)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        test_recall_list.append(test_recall)
        test_precision_list.append(test_precision)
        test_f1_list.append(test_f1)
        test_bacc_list.append(test_bacc)
        test_sp_list.append(test_sp)
        test_mcc_list.append(test_mcc)

train_df = pd.DataFrame(best_train_list, columns=['train_1A2', 'train_2C9', 'train_2C19', 'train_2D6', 'train_3A4'])
val_df = pd.DataFrame(best_val_list, columns=['valid_1A2', 'valid_2C9', 'valid_2C19', 'valid_2D6', 'valid_3A4'])
auc_df = pd.DataFrame(test_auc_list, columns=['test_1A2', 'test_2C9', 'test_2C19', 'test_2D6', 'test_3A4'])
total_auc = pd.concat([train_df, val_df, auc_df], axis=1)
total_auc.to_csv('Multi_AttentiveFP_model/10_seeds_AUC_Atten.csv', index=False)
acc_df = pd.DataFrame(test_acc_list, columns=args['task_names'])
acc_df.to_csv('Multi_AttentiveFP_model/10_seeds_ACC_Atten.csv', index=False)
recall_df = pd.DataFrame(test_recall_list, columns=args['task_names'])
recall_df.to_csv('Multi_AttentiveFP_model/10_seeds_Recall_Atten.csv', index=False)
pre_df = pd.DataFrame(test_precision_list, columns=args['task_names'])
pre_df.to_csv('Multi_AttentiveFP_model/10_seeds_precision_Atten.csv', index=False)
f1_df = pd.DataFrame(test_f1_list, columns=args['task_names'])
f1_df.to_csv('Multi_AttentiveFP_model/10_seeds_f1_score_Atten.csv', index=False)
bacc_df = pd.DataFrame(test_bacc_list, columns=args['task_names'])
bacc_df.to_csv('Multi_AttentiveFP_model/10_seeds_BACC_Atten.csv', index=False)
sp_df = pd.DataFrame(test_sp_list, columns=args['task_names'])
sp_df.to_csv('Multi_AttentiveFP_model/10_seeds_Specificity_Atten.csv', index=False)
mcc_df = pd.DataFrame(test_mcc_list, columns=args['task_names'])
mcc_df.to_csv('Multi_AttentiveFP_model/10_seeds_MCC_Atten.csv', index=False)







