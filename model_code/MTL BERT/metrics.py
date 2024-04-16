import numpy as np
import torch
from sklearn.metrics import r2_score,roc_auc_score,accuracy_score,precision_score,recall_score,confusion_matrix,f1_score,\
matthews_corrcoef,balanced_accuracy_score
import scipy

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    是一个用于追踪某个度量值的类，它会记录最近的值、平均值、总和以及计数
    """

    def __init__(self):
        self.reset()  #重置所有跟踪的指标，将它们都设为初始值

    def reset(self):
        """将以下属性设置为初始值"""
        self.val = 0 #最新的值（通常是最后一次更新的值）将被设置为 0
        self.avg = 0 #平均值将被设置为 0
        self.sum = 0 #总和将被设置为 0
        self.count = 0 #计数将被设置为 0

    def update(self, val, n=1):
        """该函数用于更新指标，在每个训练迭代中，
        我们使用 update 方法来添加新的损失值，然后通过 avg 属性获取平均值。在每个 epoch 结束时，我们打印平均损失值，然后使用 reset 方法重置 loss_meter"""
        self.val = val #要添加到指标中的新值
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# class Records_R2(object):
#     """
#     Keeps track of most recent, average, sum, and count of a metric.
#     跟踪某个指标（可能是回归任务中的R2分数），并记录每次更新的值
#     """
#     def __init__(self):
#         self.reset() #初始化对象或需要重新开始跟踪新指标时，可以调用此方法
#
#     def reset(self):
#         self.pred_list = []
#         self.label_list = []
#
#
#     def update(self, y_pred, y_true):
#         """将新的预测值 y_pred 和真实标签 y_true 添加到记录中"""
#         self.pred_list.append(y_pred)
#         self.label_list.append(y_true)
#
#     def results(self):
#         pred = np.concatenate(self.pred_list,axis=0)
#         label = np.concatenate(self.label_list,axis=0)
#
#         results = []
#         for i in range(pred.shape[1]):
#             results.append(r2_score(label[:,i]*(label[:, i]!=-1000).astype('float32'),pred[:,i],sample_weight=(label[:, i]!=-1000).astype('float32')))
#             #然后对每个列进行 R2 分数的计算。R2 分数通常用于评估回归模型的性能。这个方法返回一个包含每个列的 R2 分数的列表
#             #计算的R2分数将仅考虑非缺失值，并且缺失值（-1000）不会对分数产生影响
#
#         return results


# class Records_AUC(object):
#     """
#     Keeps track of most recent, average, sum, and count of a metric.
#     计算的是每个列的AUC（曲线下面积）分数，同时考虑到某些行可能包含缺失值（标记为-1000）的情况
#     """
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.pred_list = []
#         self.label_list = []
#
#     def update(self, y_pred, y_true):
#         self.pred_list.append(y_pred)
#         self.label_list.append(y_true)
#
#     def results(self):
#         pred = np.concatenate(self.pred_list, axis=0)
#         label = np.concatenate(self.label_list, axis=0)
#
#         results = []
#         for i in range(pred.shape[1]):
#             #它的值与原始标签数组 label 的值相同，但对于原始标签中的缺失值（-1000），将其替换为0
#             results.append(roc_auc_score((label[:, i]!=-1000)*label[:, i], pred[:, i], sample_weight=(label[:, i] != -1000).astype('float32')))
#             #第三个参数 sample_weight=(label[:, i] != -1000).astype('float32') 是样本权重，它告诉AUC分数函数哪些样本应该被考虑，哪些应该被忽略
#         return results #返回多个分类任务的auc值

class Metric(object):
    """计算模型的性能指标"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_pred = []
        self.y_true = []

    def update(self,y_pred,y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def roc_auc_score(self):
        epoch_y_pred = torch.cat(self.y_pred, axis=0)
        epoch_y_true = torch.cat(self.y_true, axis=0)
        epoch_y_prob = torch.sigmoid(epoch_y_pred)

        self.epoch_y_pred = epoch_y_pred.numpy()
        self.epoch_y_true = epoch_y_true.numpy()
        self.epoch_y_prob = epoch_y_prob.numpy()

        auc_score = []
        for i in range(self.epoch_y_true.shape[1]):
            auc_score.append(roc_auc_score((self.epoch_y_true[:, i] != -1000) * self.epoch_y_true[:, i], epoch_y_pred[:, i],
                                         sample_weight=(self.epoch_y_true[:, i] != -1000).astype('float32')))
        return auc_score

    def accuracy_score(self):
        ####创造掩码
        mask = (self.epoch_y_true != -1000)
        # 创建布尔掩码，标识哪些标签不是缺失值
        non_missing_mask = mask.astype(bool)

        acc_score = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            ###
            #print('Y_label: ',y_pred_label,'Y_prob: ',y_prob_task_filtered,'Y_pred: ',y_pred_task_filtered,
                  #'Y_true: ',y_true_task_filtered)

            # 计算任务的准确度
            accuracy_task = accuracy_score(y_true_task_filtered, y_pred_label)
            acc_score.append(accuracy_task)
        return acc_score

    def bacc(self):
        ####创造掩码
        mask = (self.epoch_y_true != -1000)
        # 创建布尔掩码，标识哪些标签不是缺失值
        non_missing_mask = mask.astype(bool)

        bacc = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            ###计算平衡准确度
            balanced_accuracy_task = balanced_accuracy_score(y_true_task_filtered,y_pred_label)
            bacc.append(balanced_accuracy_task)
        return bacc

    def specifity(self):
        mask = (self.epoch_y_true != -1000)
        non_missing_mask = mask.astype(bool)

        spe = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            conf_matrix = confusion_matrix(y_true_task_filtered,y_pred_label)
            # 从混淆矩阵计算灵敏度和特异度
            true_positive = conf_matrix[1, 1]
            false_negative = conf_matrix[1, 0]
            true_negative = conf_matrix[0, 0]
            false_positive = conf_matrix[0, 1]

            specificity = true_negative / (true_negative + false_positive)
            spe.append(specificity)
        return spe

    def recall(self):
        ####创造掩码
        mask = (self.epoch_y_true != -1000)
        # 创建布尔掩码，标识哪些标签不是缺失值
        non_missing_mask = mask.astype(bool)

        recall = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            ###计算平衡准确度
            result = recall_score(y_true_task_filtered, y_pred_label)
            recall.append(result)
        return recall

    def f1(self):
        ####创造掩码
        mask = (self.epoch_y_true != -1000)
        # 创建布尔掩码，标识哪些标签不是缺失值
        non_missing_mask = mask.astype(bool)

        f1_scores = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            ###计算平衡准确度
            result = f1_score(y_true_task_filtered, y_pred_label)
            f1_scores.append(result)
        return f1_scores

    def pre(self):
        ####创造掩码
        mask = (self.epoch_y_true != -1000)
        # 创建布尔掩码，标识哪些标签不是缺失值
        non_missing_mask = mask.astype(bool)

        precision = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            ###计算平衡准确度
            result = precision_score(y_true_task_filtered, y_pred_label)
            precision.append(result)
        return precision

    def mcc(self):
        ####创造掩码
        mask = (self.epoch_y_true != -1000)
        # 创建布尔掩码，标识哪些标签不是缺失值
        non_missing_mask = mask.astype(bool)

        mcc_scores = []
        for i in range(self.epoch_y_true.shape[1]):
            ###计算标签列表
            y_true_task = self.epoch_y_true[:, i]
            y_pred_task = self.epoch_y_pred[:, i]
            y_prob_task = self.epoch_y_prob[:, i]

            # 使用布尔掩码过滤真实标签列表、概率列表和预测值列表
            y_true_task_filtered = y_true_task[non_missing_mask[:, i]]
            y_prob_task_filtered = y_prob_task[non_missing_mask[:, i]].tolist()
            y_pred_task_filtered = y_pred_task[non_missing_mask[:, i]]

            y_pred_label = [1 if j >= 0.5 else 0 for j in y_prob_task_filtered]

            ###计算平衡准确度
            result = matthews_corrcoef(y_true_task_filtered, y_pred_label)
            mcc_scores.append(result)
        return mcc_scores








