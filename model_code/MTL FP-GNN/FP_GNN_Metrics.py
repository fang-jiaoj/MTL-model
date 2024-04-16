import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,confusion_matrix,f1_score,\
matthews_corrcoef,balanced_accuracy_score,mean_squared_error,jaccard_score

class Metric(object):
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.task_num = args.task_num  ###任务数5
        self.data_type = args.dataset_type  ###数据集类型--classification

    def auc(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)
            print('Pred_label:',pred_val_i)
            print(len(pred_val_i))
            print('True_label:',label_val_i)
            print(len(label_val_i))
            re = roc_auc_score(label_val[i], pred_val[i])  ###计算评估指标
            result.append(re)
        return result  # result 列表将包含每个任务的评估分数

    def acc(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = accuracy_score(label_val[i],pred_label_val)
            result.append(re)
        return result

    def bacc(self, pred, label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = balanced_accuracy_score(label_val[i], pred_label_val)
            result.append(re)
        return result

    def spcifity(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            conf_matrix = confusion_matrix(label_val[i],pred_label_val)
            # 从混淆矩阵计算灵敏度和特异度
            true_positive = conf_matrix[1, 1]
            false_negative = conf_matrix[1, 0]
            true_negative = conf_matrix[0, 0]
            false_positive = conf_matrix[0, 1]

            specificity = true_negative / (true_negative + false_positive)
            result.append(specificity)
        return result

    def recall(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = recall_score(label_val[i], pred_label_val)
            result.append(re)
        return result

    def f1(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = f1_score(label_val[i],pred_label_val)
            result.append(re)
        return result

    def pre(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = precision_score(label_val[i], pred_label_val)
            result.append(re)
        return result

    def mcc(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = matthews_corrcoef(label_val[i], pred_label_val)
            result.append(re)
        return result

    def jaccard(self,pred,label):
        pred_val = []
        label_val = []
        result = []
        for i in range(self.task_num):
            pred_val_i = []
            label_val_i = []
            for j in range(len(pred)):
                if label[j][i] is not None:  ###不考虑标签缺失的情况
                    pred_val_i.append(pred[j][i])  ###得到每个任务的预测结果
                    label_val_i.append(label[j][i])  ###得到每个任务的真实结果
            pred_val.append(pred_val_i)
            label_val.append(label_val_i)

            pred_label_val = [1 if t >= 0.5 else 0 for t in pred_val[i]]
            re = jaccard_score(label_val[i],pred_label_val)
            result.append(re)
        return result









