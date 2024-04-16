import torch.nn as nn
import torch.nn.functional as F

def gat_configure(args):
    gat = {
            'self_loop': False,
            'gnn_hidden_feats': 64,
            'num_layers': 2,
            'num_heads': 4,
            'predictor_hidden_feats':256,
            'dropout': 0.2454,
            'alphas': 0.414,
            'residuals': True,
            'weight_decay': 6.0322e-05,
            'lr': 0.00461,
            'batch_size': 256,
            'num_epochs': 1000,
            'patience': 30,
            'metric': 'roc_auc_score',
        }

    return gat