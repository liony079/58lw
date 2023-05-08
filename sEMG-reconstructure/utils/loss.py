import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
def nll_loss(output, target):
    return F.nll_loss(output, target).cuda()

def mse_loss(output, target):
    # loss_fn = nn.MSELoss(reduce=True, size_average=True)
    loss_fn = nn.MSELoss()
    # reduction = 'elementwise_mean'
    return loss_fn(output, target).cuda()


