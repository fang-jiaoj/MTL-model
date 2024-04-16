import pandas as pd
import os
import numpy as np
import random
from argparse import ArgumentParser
from utils import *
from utils.configure_model import gat_configure

import dgl
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping
from utils.eval_meter import Meter
from utils.featurizers import CanonicalAtomFeaturizer,CanonicalBondFeaturizer
from dgllife.utils import one_hot_encoding
from utils.gat_predictor_IG import GATPredictor
from dgllife.data.csv_dataset import MoleculeCSVDataset

###设置设备
if torch.cuda.is_available():
    print("Use GPU")
    device = 'cuda'
else:
    print("Use CPU")
    device = 'cpu'

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

###对分子图进行批处理的函数
def collate_molgraphs(data):
    assert len(data[0]) in [3,4],\
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles,graphs,labels = map(list,zip(*data))
        masks = None
    else:
        smiles,graphs,labels,masks = map(list,zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)  ###设置节点和边的初始化器为零初始化
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels,dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks,dim=0)
    return smiles,bg,labels,masks

###生成图的函数
def load_data(args,data):
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he',self_loop=args['self_loop'])
    n_feats = atom_featurizer.feat_size('hv')
    e_feats = bond_featurizer.feat_size('he')
    dataset = MoleculeCSVDataset(data,smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=args['self_loop']),
                                 node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer,smiles_column='SMILES',
                                 cache_file_path=args['data_path'] + '_graph.bin',
                                 task_names=args['task_names'],
                                 load=False,init_mask=True,n_jobs=1)
    return dataset,n_feats,e_feats

def run_a_test(args,model,data_loader):
    model.eval()
    test_meter = Meter()

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        bg = bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        test_prediction = model(bg, n_feats)
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

def test(args):
    test_data = pd.read_csv("final_external.csv")

    ###将数据处理成图
    test_set,n_feats,e_feats = load_data(args,test_data)

    ###将图数据划分成多个batch
    test_dataloader = DataLoader(dataset=test_set,batch_size=args['batch_size'],collate_fn=collate_molgraphs)

    ###定义模型
    model = GATPredictor(in_feats=n_feats,
                         hidden_feats=[args['gnn_hidden_feats']] * args['num_layers'],
                         num_heads=[args['num_heads']] * args['num_layers'],
                         feat_drops=[args['dropout']] * args['num_layers'],
                         attn_drops=[args['dropout']] * args['num_layers'],
                         alphas=[args['alphas']] * args['num_layers'],
                         residuals=[args['residuals']] * args['num_layers'],
                         predictor_hidden_feats=args['predictor_hidden_feats'],
                         predictor_dropout=args['dropout'],
                         n_tasks=5
                         )
    fn = os.path.join(args['result_path'],'MultiGAT_{}_model.pt'.format(args['seed']))
    model.load_state_dict(torch.load(fn,map_location=torch.device('cpu')))
    gat = model.to(device)

    ###进行测试
    test_auc, test_acc, test_recall, test_precision, test_f1, test_bacc, test_sp, test_mcc = \
        run_a_test(args, gat, test_dataloader)
    test_score_avg = np.mean(test_auc)
    print('Test {} {:.4f}'.format(args['metric'], test_score_avg))  # 计算测试集的性能指标
    if args['n_tasks'] > 1:
        for one_name, one_score in zip(args['task_names'], test_auc):
            print('Test {} {} = {:.4f}'.format(one_name, args['metric'], one_score))

    return test_auc, test_acc, test_recall, test_precision, test_f1, test_bacc, test_sp, test_mcc

if __name__ == '__main__':
    parser = ArgumentParser(description='GAT Prediction') ###创建一个参数解析器
    parser.add_argument('--data_path',type=str,default='Multi_GAT_data/')
    parser.add_argument('--result_path',type=str,default='Multi_GAT_model/')
    parser.add_argument('--task_names',type=str,default='1A2,2C9,2C19,2D6,3A4',
                        help='The name of tasks.')
    parser.add_argument('--seed',type=int,default=0,help='The seed of random.')
    parser.add_argument('--n_tasks',type=int,default=5,help='The number of tasks.')
    parser.add_argument('--num_folds',type=int,default=1,
                   help='The number of folds in cross validation.')

    args = parser.parse_args().__dict__
    if args['task_names'] is not None:
        args['task_names'] = args['task_names'].split(',') ###使用逗号分隔开
    args.update(gat_configure(args))
    set_random_seed(args)

    seed_first = args['seed']
    test_auc_list = []
    test_acc_list = []
    test_recall_list = []
    test_precision_list = []
    test_f1_list = []
    test_bacc_list = []
    test_sp_list = []
    test_mcc_list = []
    for i in range(10):
        args['seed'] = seed_first + i
        print('Seed {}'.format(args['seed']))
        test_auc, test_acc, test_recall, test_precision, test_f1, test_bacc, test_sp, test_mcc = test(args)
        test_auc = np.array(test_auc).reshape((1,5))
        test_acc = np.array(test_acc).reshape((1,5))
        test_bacc = np.array(test_bacc).reshape(1,5)
        test_recall = np.array(test_recall).reshape((1,5))
        test_sp = np.array(test_sp).reshape((1,5))
        test_precision = np.array(test_precision).reshape((1,5))
        test_f1 = np.array(test_f1).reshape((1,5))
        test_mcc = np.array(test_mcc).reshape((1,5))

        test_auc_list.extend(test_auc)
        test_acc_list.extend(test_acc)
        test_recall_list.extend(test_recall)
        test_precision_list.extend(test_precision)
        test_f1_list.extend(test_f1)
        test_bacc_list.extend(test_bacc)
        test_sp_list.extend(test_sp)
        test_mcc_list.extend(test_mcc)

    auc_df = pd.DataFrame(test_auc_list, columns=args['task_names'])
    auc_df.to_csv('Multi_GAT_model/Ext_10_seeds_AUC_GAT.csv', index=False)
    acc_df = pd.DataFrame(test_acc_list, columns=args['task_names'])
    acc_df.to_csv('Multi_GAT_model/Ext_10_seeds_ACC_GAT.csv', index=False)
    recall_df = pd.DataFrame(test_recall_list, columns=args['task_names'])
    recall_df.to_csv('Multi_GAT_model/Ext_10_seeds_Recall_GAT.csv', index=False)
    pre_df = pd.DataFrame(test_precision_list, columns=args['task_names'])
    pre_df.to_csv('Multi_GAT_model/Ext_10_seeds_precision_GAT.csv', index=False)
    f1_df = pd.DataFrame(test_f1_list, columns=args['task_names'])
    f1_df.to_csv('Multi_GAT_model/Ext_10_seeds_f1_score_GAT.csv', index=False)
    bacc_df = pd.DataFrame(test_bacc_list, columns=args['task_names'])
    bacc_df.to_csv('Multi_GAT_model/Ext_10_seeds_BACC_GAT.csv', index=False)
    sp_df = pd.DataFrame(test_sp_list, columns=args['task_names'])
    sp_df.to_csv('Multi_GAT_model/Ext_10_seeds_Specificity_GAT.csv', index=False)
    mcc_df = pd.DataFrame(test_mcc_list, columns=args['task_names'])
    mcc_df.to_csv('Multi_GAT_model/EXT_10_seeds_MCC_GAT.csv', index=False)


