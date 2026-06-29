import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure


## Loss function

def f1_loss(pred, target):
    pred = torch.sigmoid(pred)

    tp = torch.sum(pred * target)
    tn = torch.sum((1-pred) * (1-target))
    fp = torch.sum((1-pred) * target)
    fn = torch.sum(pred * (1 - target))

    eps = torch.from_numpy(np.asarray(torch.finfo(torch.float32).eps))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = 2 * p * r / (p + r + eps)
    f1[torch.isnan(f1)] = 0
    f1[torch.isinf(f1)] = 0
    return 1 - f1

def smoothiouLoss(pred, target, smooth=1e-12):
    """
    Calculate the Intersection over Union (IoU) loss.

    Args:
    - pred (torch.Tensor): The predicted segmentation map. Expected shape is (N, 1, H, W).
    - target (torch.Tensor): The ground truth segmentation map. Expected shape is (N, 1, H, W).
    - smooth (float): A small smoothing value to avoid division by zero.

    Returns:
    - torch.Tensor: The IoU loss.
    """
    pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities if `pred` is logits

    intersection = (pred * target).sum(dim=[1, 2, 3])  # Intersection part
    union = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) - intersection  # Union part

    iou = (intersection + smooth) / (union + smooth)  # IoU calculation
    loss = 1 - iou  # IoU loss

    return loss.mean()  # Return the mean IoU loss for the batch

def siouLoss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6  # Small value for numerical stability

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    # intersection = (pred * target).sum(dim=(2, 3))

    softiou = (intersection + smooth) / (union + smooth)

    gt_siou = (target.sum(dim=(1, 2, 3)) + smooth) / ((target > 0).sum(dim=(1, 2, 3)) + smooth)
    softiou /= gt_siou

    loss = 1 - softiou.mean()

    return loss

def sflLoss(preds, targets, gamma=2.0):
    preds = torch.sigmoid(preds)
    
    epsilon = 1e-6  # Small value for numerical stability
    preds = torch.clamp(preds, epsilon, 1.0 - epsilon)
    targets = torch.clamp(targets, 0.0, 1.0)

    pt = (1 - targets) * torch.log(1 - preds) + targets * torch.log(preds)
    w = torch.pow(torch.abs(targets - preds), gamma)
    loss = -w * pt
    # print(f"loss shape: {loss.shape}, dtypesfl: {loss.dtype}")

    return loss.mean()

def Dice(pred, target, warm_epoch=1, epoch=1, layer=0):
    pred = torch.sigmoid(pred)

    smooth = 1

    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    loss = (2 * intersection_sum + smooth) / \
           (pred_sum + target_sum + intersection_sum + smooth)

    loss = 1 - loss.mean()

    return loss

def soft_dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Apply sigmoid if pred is logits

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

#SLS Loss
def SoftIoULoss(pred, target):
    pred = torch.sigmoid(pred)

    smooth = 1

    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    loss = (intersection_sum + smooth) / \
            (pred_sum + target_sum - intersection_sum + smooth)

    loss = 1 - loss.mean()

    return loss

def SLSIoULoss(pred_log, target, warm_epoch, epoch, with_shape=True):
    pred = torch.sigmoid(pred_log)
    smooth = 0.0

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    dis = torch.pow((pred_sum - target_sum) / 2, 2)

    alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

    loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
    lloss = LLoss(pred, target)

    if epoch > warm_epoch:
        siou_loss = alpha * loss
        if with_shape:
            loss = 1 - siou_loss.mean() + lloss
        else:
            loss = 1 - siou_loss.mean()
    else:
        loss = 1 - loss.mean()
    return loss

def SLoss(pred_log, target, warm_epoch, epoch, with_shape=True):
    pred = torch.sigmoid(pred_log)
    smooth = 0.0

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))

    # dis = torch.pow((pred_sum - target_sum) / 2, 2)

    alpha = (torch.min(pred_sum, target_sum) + smooth) / (torch.max(pred_sum, target_sum) + smooth)

    loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
    # lloss = LLoss(pred, target)

    if epoch > warm_epoch:
        siou_loss = alpha * loss
        # if with_shape:
        #     loss = 1 - siou_loss.mean() + lloss
        # else:
        #     loss = 1 - siou_loss.mean()
        loss = 1 - siou_loss.mean()
    else:
        loss = 1 - loss.mean()
    return loss

def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth))
                                                           - torch.arctan(
            (target_centery) / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss

def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)
    pred = torch.sigmoid(pred)

    pos_inds = target.gt(0.8).float()
    neg_inds = target.lt(0.5).float()

    neg_weights = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * 100
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * 0.1

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def FocalLoss(inputs, target, alpha=0.75, gamma=2, reduction='mean'):
    p = inputs

    bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, target, reduction='none')

    alpha = alpha * target + (1 - alpha) * (1 - target)
    focal_loss = alpha * ((1 - p) ** gamma) * bce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

def l1_loss(pred, target, mask=None):

    if mask is not None:
        mask = mask.unsqueeze(1).expand_as(pred)  # [batch, 2, H, W]
        loss = torch.abs(pred - target) * mask.float()
        return loss.sum() / (mask.sum() + 1e-4)
    else:
        return nn.L1Loss()(pred, target)

def HuberLoss(pred, target, delta=1):
    pred = torch.sigmoid(pred)
    diff = torch.abs(target - pred)
    loss = torch.where(diff < delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    return torch.mean(loss)


def offset_loss(self, pred_probs, target_dist):
    # target_dist: tuple of (x_target_dist, y_target_dist)
    x_prob, y_prob = pred_probs
    x_target, y_target = target_dist

    # Cross entropy loss
    ce_loss = F.binary_cross_entropy(x_prob, x_target) + \
                F.binary_cross_entropy(y_prob, y_target)

    # Entropy regularization
    entropy = -(x_prob * torch.log(x_prob + 1e-9)).mean() + \
                  -(y_prob * torch.log(y_prob + 1e-9)).mean()

    return ce_loss + 0.1 * entropy

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count