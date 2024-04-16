import csv
import numpy as np
import os
from fpgnn_ext.tool import set_predict_argument, get_scaler, load_args, load_data, load_model,get_task_name
from fpgnn_ext.train import predict
from fpgnn_ext.data import MoleDataSet
from FP_GNN_Metrics import Metric
import pandas as pd

def predicting(args):
    """这段代码主要用于将训练好的模型应用于新的数据集以进行预测，并将预测结果保存到输出文件中，
    这段代码将预测结果和分子 SMILES 写入一个 CSV 文件，该文件包含了分子 SMILES 以及每个任务的预测结果。如果没有有效的预测结果，相应位置将被填充为空字符串"""
    print('Load args.')
    scaler = get_scaler(args.model_path)
    print('scaler',scaler)  ###导入模型的参数
    train_args = load_args(args.model_path) #从指定的模型路径 args.model_path 中加载用于训练模型时的超参数

    for key,value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
            #如果当前参数 key 还未添加到 args 对象中，那么代码使用 setattr 函数将其添加，
            # 并将其值设置为 value。这样可以确保在进行预测时，所有必要的参数都已经准备好，并且与训练时的参数一致

    print('Load data.')
    test_data = load_data(args.predict_path,args) ###生成包含所有测试数据的有效的MoleDatasets
    fir_data_len = len(test_data)
    all_data = test_data
    if fir_data_len == 0:
        raise ValueError('Data is empty.')

    #处理测试数据集，特别是检查数据中是否存在无效的记录（无效的记录可能是由于某些样本的分子结构无法成功解析而导致的）并进行相关的操作
    smi_exist = []
    for i in range(fir_data_len):
        if test_data[i].mol is not None:
            smi_exist.append(i)
    test_data = MoleDataSet([test_data[i] for i in smi_exist])  ###得到有效的MoleDataSet,它里面包括了很多MoleData
    now_data_len = len(test_data)
    print('There are ',now_data_len,' smiles in total.')
    if fir_data_len - now_data_len > 0:
        print('There are ',fir_data_len - now_data_len, ' smiles invalid.')

    ###负责加载模型并用它对测试数据集进行预测
    print('Load model')
    model = load_model(args.model_path,args.cuda) ###加载训练好的模型参数，并把模型放到GPU上
    test_pred = predict(model,test_data,args.batch_size,scaler) #使用加载的模型对测试数据集 test_data 进行预测,输出预测概率
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    print('Test_pred:',test_pred,len(test_pred))

    print("Calculate metrics")   ###注意：不考虑缺失值
    test_label = test_data.label() ###数据的真实标签
    print('Test_true:',test_label,len(test_label))
    metrics = Metric(train_args)
    auc_score = np.array(metrics.auc(test_pred,test_label)).reshape(1,5)
    acc_socre = np.array(metrics.acc(test_pred,test_label)).reshape(1,5)
    bacc_score = np.array(metrics.bacc(test_pred,test_label)).reshape(1,5)
    sp = np.array(metrics.spcifity(test_pred,test_label)).reshape(1,5)
    re = np.array(metrics.recall(test_pred,test_label)).reshape(1,5)
    f1_score = np.array(metrics.f1(test_pred,test_label)).reshape(1,5)
    pre = np.array(metrics.pre(test_pred,test_label)).reshape(1,5)
    mcc_score = np.array(metrics.mcc(test_pred,test_label)).reshape(1,5)
    jaccard = np.array(metrics.jaccard(test_pred,test_label)).reshape(1,5)




    # auc_df = pd.DataFrame(auc_score.reshape((1, 1)), index=['1A2'], columns=['AUC'])
    # acc_df = pd.DataFrame(acc_socre.reshape((1, 1)), index=['1A2'], columns=['ACC'])
    # bacc_df = pd.DataFrame(bacc_score.reshape((1, 1)), index=['1A2'], columns=['BACC'])
    # sp_df = pd.DataFrame(sp.reshape((1, 1)), index=['Specificity'], columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    # re_df = pd.DataFrame(re.reshape((1, 1)), index=['Recall'], columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    # f1_df = pd.DataFrame(f1_score.reshape((1, 5)), index=['F1_score'], columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    # pre_df = pd.DataFrame(pre.reshape((1, 5)), index=['Precision'], columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    # mcc_df = pd.DataFrame(mcc_score.reshape((1, 5)), index=['MCC'], columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    # result_total = pd.concat([auc_df,acc_df, bacc_df,sp_df,re_df, f1_df, pre_df, mcc_df], axis=0)
    # result_total.to_csv(r'model_save_random/Seed_9/Seed_9_test.csv')

    if args.task_num > 1:
        args.task_names = get_task_name(args.predict_path)
        for num, clf_head in enumerate(args.task_names):
            print('test auc {}: {:.4f}'.format(clf_head, auc_score[num]))  ##输出一个epoch每个任务的AUC值
            print('test acc {}: {:.4f}'.format(clf_head, acc_socre[num]))
            print('test bacc {}: {:.4f}'.format(clf_head, bacc_score[num]))
            print('test specifity {}: {:.4f}'.format(clf_head, sp[num]))
            print('test recall {}: {:.4f}'.format(clf_head, re[num]))
            print('test f1_score {}: {:.4f}'.format(clf_head, f1_score[num]))
            print('test precision {}: {:.4f}'.format(clf_head, pre[num]))
            print('test mcc_score {}: {:.4f}'.format(clf_head, mcc_score[num]))

    return auc_score,acc_socre,bacc_score,sp,re,f1_score,pre,mcc_score,jaccard

if __name__=='__main__':
    args = set_predict_argument()
    seed_first = 0
    model_path = args.model_path
    test_auc_list = []
    test_acc_list = []
    test_recall_list = []
    test_precision_list = []
    test_f1_list = []
    test_bacc_list = []
    test_sp_list = []
    test_mcc_list = []
    test_jaccard_list = []
    for seed in range(10):
        args.seed = seed + seed_first
        model_dir = os.path.join(model_path,f"Seed_{args.seed}")
        args.model_path = os.path.join(model_dir,f'{args.seed}_model.pt')
        args.predict_path = 'external_2.csv'
        test_auc, test_acc,test_bacc, test_sp, test_recall, test_f1,test_precision, test_mcc,jaccard = predicting(args)
        test_auc_list.extend(test_auc)
        test_acc_list.extend(test_acc)
        test_bacc_list.extend(test_bacc)
        test_sp_list.extend(test_sp)
        test_recall_list.extend(test_recall)
        test_f1_list.extend(test_f1)
        test_precision_list.extend(test_precision)
        test_mcc_list.extend(test_mcc)
        test_jaccard_list.extend(jaccard)

    auc_df = pd.DataFrame(test_auc_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    auc_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_AUC_FP-GNN.csv", index=False)

    acc_df = pd.DataFrame(test_acc_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    acc_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_ACC_FP-GNN.csv", index=False)

    bacc_df = pd.DataFrame(test_bacc_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    bacc_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_BACC_FP-GNN.csv", index=False)

    sp_df = pd.DataFrame(test_sp_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    sp_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_SP_FP-GNN.csv", index=False)

    recall_df = pd.DataFrame(test_recall_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    recall_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_Recall_FP-GNN.csv", index=False)

    f1_df = pd.DataFrame(test_f1_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    f1_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_F1_score_FP-GNN.csv", index=False)

    pre_df = pd.DataFrame(test_precision_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    pre_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_Precision_FP-GNN.csv", index=False)

    mcc_df = pd.DataFrame(test_mcc_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    mcc_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_MCC_FP-GNN.csv", index=False)

    jaccard_df = pd.DataFrame(test_jaccard_list,columns=['1A2', '2C9', '2C19', '2D6', '3A4'])
    jaccard_df.to_csv("FPGNN_model_save_random_最终结果/Ext2_10_seeds_jaccard_FP-GNN.csv", index=False)
