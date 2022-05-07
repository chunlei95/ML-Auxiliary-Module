import torch.nn as nn


def cross_entropy(predict, target):
    """交叉熵损失

    :param predict: 预测值
    :param target: 目标值（即标签）
    :return: 预测值与目标值之间的损失值
    """
    ce = nn.CrossEntropyLoss()
    return ce(predict, target)
