import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_loss(pred, target, loss_type):
    if loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'smoothL1':
        loss = F.smooth_l1_loss(pred, target)
    elif loss_type == 'binary':
        loss = F.binary_cross_entropy(pred, target)
    else:
        raise NotImplemented('wrong loss type')

    return loss
