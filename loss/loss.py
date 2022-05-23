import torch
import torch.nn as nn
import sklearn.metrics as metrics


def cross_entropy(predict, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean',
                  label_smoothing=0.0):
    """交叉熵损失，用于多分类问题

    :param label_smoothing:
    :param ignore_index:
    :param reduction:
    :param reduce:
    :param size_average:
    :param weight:
    :param predict: 预测值
    :param target: 目标值（即标签）
    :return: 预测值与目标值之间的损失值
    """
    # return metrics.log_loss(target, predict)
    ce = nn.CrossEntropyLoss(weight, size_average, ignore_index=ignore_index, reduction=reduction, reduce=reduce,
                             label_smoothing=label_smoothing)
    return ce(predict, target)


def bce_loss(predict, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    """二值交叉熵损失函数，用于二分类问题中

    :param reduction:
    :param reduce:
    :param weight:
    :param size_average:
    :param predict: 预测值
    :param target: 真实值
    :return: 二值交叉熵损失值
    """
    bce = nn.BCELoss(weight, size_average, reduce, reduction)
    return bce(predict, target)


class DiceLoss(nn.Module):

    def __init__(self, ep=1e-8):
        super(DiceLoss, self).__init__()
        self.epsilon = ep

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be same"
        num = predict.size()
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score


def dice_loss(predict, target, ep=1e-8):
    """dice相似系数损失函数

    :param predict: 预测值
    :param target: 真实值
    :param ep: 平滑系数
    :return:
    """
    intersection = 2 * torch.sum(predict * target) + ep
    union = torch.sum(predict) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss
