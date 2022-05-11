import torch.utils.data as data


def trainer(train_data, network_model, optimizer, loss, hyper_parameters, device=None):
    """

    :param train_data: 训练数据
    :param network_model: 要训练的模型
    :param optimizer: 使用的模型优化器
    :param loss: 使用的损失函数
    :param hyper_parameters: 容纳模型训练超参数的列表
    :param device:
    :return: 训练后的模型
    """
    epoch = hyper_parameters.get('epoch') if hyper_parameters.get('epoch') is not None else 20
    batch_size = hyper_parameters.get('batch_size') if hyper_parameters.get('batch_size') is not None else 1
    num_workers = hyper_parameters.get('num_workers') if hyper_parameters.get('num_workers') is not None else 2
    train_data = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for step in range(epoch):
        for i, (x, y) in enumerate(train_data):
            x = x.to(device)
            y = y.to(device)
            y_predict = network_model(x)
            loss_value = loss(y_predict, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
    return network_model
