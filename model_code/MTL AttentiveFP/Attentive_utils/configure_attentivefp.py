import torch.nn as nn
import torch.nn.functional as F

def attentivefp_configure(args):
    attentivefp = {
        'self_loop': False,
        'graph_feat_size': 64,
        'num_layers': 2,
        'num_timesteps': 1,
        'predictor_hidden_feats': 256,
        'dropout': 0.26,
        'weight_decay': 4.1582e-05,
        'lr': 0.00095,
        'batch_size': 256,
        'num_epochs': 1000,
        'patience': 25,
        'metric': 'roc_auc_score',
    }

    return attentivefp