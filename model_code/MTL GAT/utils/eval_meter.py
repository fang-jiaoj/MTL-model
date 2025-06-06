# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluation of model performance."""
# pylint: disable= no-member, arguments-differ, invalid-name

import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc ,accuracy_score, \
    recall_score, precision_score, f1_score, roc_curve,matthews_corrcoef,balanced_accuracy_score,confusion_matrix


__all__ = ['Meter']

# pylint: disable=E1101
class Meter(object):
    """Track and summarize model performance on a dataset for (multi-label) prediction.

    When dealing with multitask learning, quite often we normalize the labels so they are
    roughly at a same scale. During the evaluation, we need to undo the normalization on
    the predicted labels. If mean and std are not None, we will undo the normalization.

    Currently we support evaluation with 4 metrics:

    * ``pearson r2``
    * ``mae``
    * ``rmse``
    * ``roc auc score``

    Parameters
    ----------
    mean : torch.float32 tensor of shape (T) or None.
        Mean of existing training labels across tasks if not ``None``. ``T`` for the
        number of tasks. Default to ``None`` and we assume no label normalization has been
        performed.
    std : torch.float32 tensor of shape (T)
        Std of existing training labels across tasks if not ``None``. Default to ``None``
        and we assume no label normalization has been performed.

    Examples
    --------
    Below gives a demo for a fake evaluation epoch.

    >>> import torch
    >>> from dgllife.utils import Meter

    >>> meter = Meter()
    >>> # Simulate 10 fake mini-batches
    >>> for batch_id in range(10):
    >>>     batch_label = torch.randn(3, 3)
    >>>     batch_pred = torch.randn(3, 3)
    >>>     meter.update(batch_pred, batch_label)

    >>> # Get MAE for all tasks
    >>> print(meter.compute_metric('mae'))
    [1.1325558423995972, 1.0543707609176636, 1.094650149345398]
    >>> # Get MAE averaged over all tasks
    >>> print(meter.compute_metric('mae', reduction='mean'))
    1.0938589175542195
    >>> # Get the sum of MAE over all tasks
    >>> print(meter.compute_metric('mae', reduction='sum'))
    3.2815767526626587
    """
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        """获得预测值，真实值和掩码"""
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted labels with shape ``(B, T)``,
            ``B`` for number of graphs in the batch and ``T`` for the number of tasks
        y_true : float32 tensor
            Ground truth labels with shape ``(B, T)``
        mask : None or float32 tensor
            Binary mask indicating the existence of ground truth labels with
            shape ``(B, T)``. If None, we assume that all labels exist and create
            a one-tensor for placeholder.
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(torch.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        """用于在进行模型性能评估之前做准备工作"""
        """Prepare for evaluation.

        If normalization was performed on the ground truth labels during training,
        we need to undo the normalization on the predicted labels.

        Returns
        -------
        mask : float32 tensor
            Binary mask indicating the existence of ground
            truth labels with shape (B, T), B for batch size
            and T for the number of tasks
        y_pred : float32 tensor
            Predicted labels with shape (B, T)
        y_true : float32 tensor
            Ground truth labels with shape (B, T)
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            # To compensate for the imbalance between labels during training,
            # we normalize the ground truth labels with training mean and std.
            # We need to undo that for evaluation.
            y_pred = y_pred * self.std + self.mean

        ###二进制掩码 mask、模型预测的标签 y_pred 和真实标签 y_true
        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        """根据指定的 reduction 操作对任务的得分进行汇总"""
        """Finalize the scores to return.

        Parameters
        ----------
        scores : list of float
            Scores for all tasks.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        """Evaluate for multi-label prediction.

        Parameters
        ----------
        score_func : callable
            A score function that takes task-specific ground truth and predicted labels as
            input and return a float as the score. The labels are in the form of 1D tensor.
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        mask, y_pred, y_true = self._finalize() ###获取二进制掩码 mask、预测标签 y_pred 和真实标签 y_true。
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task] ###提取对应的掩码
            task_y_true = y_true[:, task][task_w != 0]  ###非缺失值的真实标签
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score) ###计算任务得分
        return self._reduce_scores(scores, reduction) ###对所有任务的得分进行适当的汇总

    def pearson_r2(self, reduction='none'):
        """Compute squared Pearson correlation coefficient.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0] ** 2
        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        """Compute mean absolute error.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()
        return self.multilabel_score(score, reduction)

    def rmse(self, reduction='none'):
        """Compute root mean square error.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        def score(y_true, y_pred):
            return torch.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()
        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        """用于计算二分类任务的 ROC-AUC 分数"""
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        ###这是一个断言语句，用于确保标签没有经过规范化(标准化)
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            ###这个条件语句检查真实标签 y_true 中是否只包含一个类别
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                ###如果任务有多个类别，调用 roc_auc_score 函数计算二分类任务的 ROC-AUC 分数
                return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
        return self.multilabel_score(score, reduction) ###返回最终的AUC值

    def pr_auc_score(self, reduction='none'):
        """Compute the area under the precision-recall curve (pr-auc score)
        for binary classification.

        PR-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case, we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
                return auc(recall, precision)
        return self.multilabel_score(score, reduction)

    def accuracy_score(self, reduction='none'):
        """用于计算二分类任务的准确率（accuracy）得分的方法"""
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        ##这是一个断言语句，用于确保标签没有经过规范化（标准化）
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            ###条件语句检查真实标签 y_true 中是否只包含一个类别
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'precision score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                ###将真实标签 y_true 转换为 NumPy 数组，并将其从 GPU 移动到 CPU。然后将其转换为 Python 列表。
                y_true = y_true.detach().cpu().long().numpy().tolist()
                ###将预测标签 y_pred 通过 sigmoid 函数进行概率化，并将其从 GPU 移动到 CPU。然后将其转换为 NumPy 数组、扁平化并转换为 Python 列表。
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                ###最终的预测标签 y_pred
                y_pred = np.array(pred_y)
                return accuracy_score(y_true, y_pred)  ###计算真实值和预测标签的ACC值
        return self.multilabel_score(score, reduction) ###返回所有任务的准确率得分

    def precision_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'precision score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return precision_score(y_true, y_pred)
        return self.multilabel_score(score, reduction)


    def recall_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'Recall score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return recall_score(y_true, y_pred)
        return self.multilabel_score(score, reduction)

    def f1_score(self, reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
        for binary classification.

        ROC-AUC scores are not well-defined in cases where labels for a task have one single
        class only (e.g. positive labels only or negative labels only). In this case we will
        simply ignore this task and print a warning message.

        Parameters
        ----------
        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks.

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'
        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'Recall score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return f1_score(y_true, y_pred)
        return self.multilabel_score(score, reduction)

    def bacc(self,reduction='none'):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
                for binary classification.

                ROC-AUC scores are not well-defined in cases where labels for a task have one single
                class only (e.g. positive labels only or negative labels only). In this case we will
                simply ignore this task and print a warning message.

                Parameters
                ----------
                reduction : 'none' or 'mean' or 'sum'
                    Controls the form of scores for all tasks.

                Returns
                -------
                float or list of float
                    * If ``reduction == 'none'``, return the list of scores for all tasks.
                    * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                    * If ``reduction == 'sum'``, return the sum of scores for all tasks.
                """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'Recall score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return balanced_accuracy_score(y_true, y_pred)
        return self.multilabel_score(score, reduction)

    def specifity(self,reduction=None):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
                    for binary classification.

                    ROC-AUC scores are not well-defined in cases where labels for a task have one single
                    class only (e.g. positive labels only or negative labels only). In this case we will
                    simply ignore this task and print a warning message.

                    Parameters
                    ----------
                    reduction : 'none' or 'mean' or 'sum'
                        Controls the form of scores for all tasks.

                    Returns
                    -------
                    float or list of float
                    * If ``reduction == 'none'``, return the list of scores for all tasks.
                    * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                    * If ``reduction == 'sum'``, return the sum of scores for all tasks.
                        """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        mask, y_pred, y_true = self._finalize()  ###获取二进制掩码 mask、预测标签 y_pred 和真实标签 y_true。
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]  ###提取对应的掩码
            task_y_true = y_true[:, task][task_w != 0].detach().cpu().long().numpy().tolist()  ###非缺失值的真实标签
            task_y_pred = y_pred[:, task][task_w != 0]
            task_y_prob = torch.sigmoid(task_y_pred).detach().cpu().numpy().flatten().tolist()
            task_y_pred_label = [1 if i >= 0.5 else 0 for i in task_y_prob]

            conf_matrix = confusion_matrix(task_y_true, task_y_pred_label)
            # 从混淆矩阵计算灵敏度和特异度
            true_positive = conf_matrix[1, 1]
            false_negative = conf_matrix[1, 0]
            true_negative = conf_matrix[0, 0]
            false_positive = conf_matrix[0, 1]

            specificity = true_negative / (true_negative + false_positive)
            scores.append(specificity)
        return self._reduce_scores(scores, reduction)  ###对所有任务的得分进行适当的汇总

    def mcc(self,reduction=None):
        """Compute the area under the receiver operating characteristic curve (roc-auc score)
                for binary classification.

                ROC-AUC scores are not well-defined in cases where labels for a task have one single
                class only (e.g. positive labels only or negative labels only). In this case we will
                simply ignore this task and print a warning message.

                Parameters
                ----------
                reduction : 'none' or 'mean' or 'sum'
                    Controls the form of scores for all tasks.

                Returns
                -------
                float or list of float
                    * If ``reduction == 'none'``, return the list of scores for all tasks.
                    * If ``reduction == 'mean'``, return the mean of scores for all tasks.
                    * If ``reduction == 'sum'``, return the sum of scores for all tasks.
                """
        # Todo: This function only supports binary classification and we may need
        #  to support categorical classes.
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'Recall score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                y_true = y_true.detach().cpu().long().numpy().tolist()
                probs_y = torch.sigmoid(y_pred).detach().cpu().numpy().flatten().tolist()
                pred_y = []
                for i in probs_y:
                    if i >= 0.5:
                        i = 1
                        pred_y.append(i)
                    if i < 0.5:
                        i = 0
                        pred_y.append(i)
                y_pred = np.array(pred_y)
                return matthews_corrcoef(y_true, y_pred)
        return self.multilabel_score(score, reduction)



    def compute_metric(self, metric_name, reduction='none'):
        """用于根据给定的性能指标名称计算性能指标的方法,根据 metric_name 的不同取值，方法将调用相应的性能评估方法"""
        """Compute metric based on metric name.

        Parameters
        ----------
        metric_name : str

            * ``'r2'``: compute squared Pearson correlation coefficient
            * ``'mae'``: compute mean absolute error
            * ``'rmse'``: compute root mean square error
            * ``'roc_auc_score'``: compute roc-auc score
            * ``'pr_auc_score'``: compute pr-auc score

        reduction : 'none' or 'mean' or 'sum'
            Controls the form of scores for all tasks

        Returns
        -------
        float or list of float
            * If ``reduction == 'none'``, return the list of scores for all tasks.
            * If ``reduction == 'mean'``, return the mean of scores for all tasks.
            * If ``reduction == 'sum'``, return the sum of scores for all tasks.
        """
        if metric_name == 'r2':
            return self.pearson_r2(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        elif metric_name == 'precision':
            return self.precision_score(reduction)
        elif metric_name == 'recall':
            return self.recall_score(reduction)
        elif metric_name == 'accuracy':
            return self.accuracy_score(reduction)
        elif metric_name == 'f1':
            return self.f1_score(reduction)
        elif metric_name == 'bacc':
            return self.bacc(reduction)
        elif metric_name == 'sp':
            return self.specifity(reduction)
        elif metric_name == 'mcc':
            return self.mcc(reduction)
        else:
            raise ValueError('Expect metric_name to be "r2" or "mae" or "rmse" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))
