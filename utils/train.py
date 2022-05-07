def trainer(model, optimizer, loss, hyper_parameters, train_data):
    """

    :param model: 要训练的模型
    :param optimizer: 使用的模型优化器
    :param loss: 使用的损失函数
    :param hyper_parameters: 容纳模型训练超参数的列表
    :param train_data: 要训练的数据
    :return:
    """
    epoch = hyper_parameters.get('epoch')
    lr = hyper_parameters.get('lr')
    if epoch is None:
        epoch = 1
    if lr is None:
        lr = 1e-5
    pass
