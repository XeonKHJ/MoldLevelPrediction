from loss.dilate_loss import dilate_loss
import torch

def exeDialoteLoss(x, y):
    loss, temp, shape = dilate_loss(x, y, 0.5, 0.001, torch.device('cuda'))
    return loss

def DialteLoss():
    return (lambda x, y: exeDialoteLoss(x, y))

