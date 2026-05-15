import torch
import torch.nn as nn

def get_pos_weights(y_train):

    y = torch.tensor(y_train)

    pos_counts = y.sum(dim=0)
    neg_counts = len(y) - pos_counts

    weights = neg_counts / (pos_counts + 1e-6)

    return weights


def create_weighted_loss(y_train):

    pos_weights = get_pos_weights(y_train)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights
    )

    return criterion