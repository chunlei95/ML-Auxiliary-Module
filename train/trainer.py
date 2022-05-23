import uuid
import torch

import sklearn.metrics as metrics


def trainer(model, datasets, criterion, optimizer, *, params: dict = None, type='classify'):
    """单个模型，单个损失函数，单个优化器的训练

    :param model: 模型
    :param datasets: 训练数据，datasets的形式为(x, y)的元组，x为训练集的输入，y为训练集的标签，
                    对于无标注数据，需要设置一个占位符来填充y。datasets需要提前在外面做好预处理
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param params: 字典形式的超参数，包括epoch, lr, batch_size, epoch的默认值为100，
                    lr的默认值为1e-4，batch_size的默认值为64
    :param type: 任务类型，① classify：分类任务，默认值 ② semantic：语义分割任务 ③ regression：回归任务
    :return:
    """
    epoch = params.get('epoch') if params.get('epoch') is not None else 100
    task_name = params.get('task_name') if params.get('task_name') else uuid.UUID()
    save_params = params.get('save_params') if params.get('save_params') else False
    pth_path = params.get('pth_path') if params.get('pth_path') else './model_params/' + task_name
    device = params.get('device') if params.get('device') else 'cpu'
    model.to(device)
    optimizer.to(device)
    avg_loss_changes = []
    last_loss_changes = []
    for i in range(epoch):
        loss_total, loss, sample_count, accuracy, idx = 0.0, 0.0, 0, 0.0, 0
        for index, (x, y) in enumerate(datasets):
            y_predict = model(x.to(device))
            loss = criterion(y.to(device), y_predict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss
            sample_count += len(y)
            accuracy = metrics.accuracy_score(y, y_predict)
            idx = index
        last_loss_changes.append(loss)
        avg_loss = loss_total / sample_count
        avg_loss_changes.append(avg_loss)
        if type == 'classify':
            print('epoch {}: Batch {}/{} average loss = {:.4f} last loss = {:.4f} accuracy = {:.4f}'
                  .format(i, idx, len(datasets), avg_loss, loss, accuracy))
        elif type == 'semantic':
            pass
        elif type == 'regression':
            pass
    if save_params:
        params = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'average_loss_change': avg_loss_changes,
            'last_loss_change': last_loss_changes
        }
        torch.save(params, pth_path)
    return model


def test(model, model_path, data, *, metric: list = None, params: dict = None):
    """模型推理

    :param model: 训练好的模型
    :param model_path: 模型训练完后参数的保存路径
    :param data: 测试数据集
    :param metric: 指定使用哪些评价指标
    :param params: 其它参数
    :return: 评价指标的结果集合，与metric对应
    """
    model_params = torch.load(model_path)
    model.load_state_dict(model_params['model_state_dict'])
    model.eval()
    x, y = data[0], data[1]
    y_predict = model(x)
    metric_dict = {}
    if 'accuracy' in metric:
        metric_dict['accuracy'] = metrics.accuracy_score(y, y_predict)
    if 'precision' in metric:
        metric_dict['precision'] = metrics.precision_score(y, y_predict)
    if 'recall' in metric:
        metric_dict['recall'] = metrics.recall_score(y, y_predict)
    if 'f1' in metric:
        metric_dict['f1'] = metrics.f1_score(y, y_predict)
    if 'jaccard' in metric:
        metric_dict['jaccard'] = metrics.jaccard_score(y, y_predict)
    if 'auc' in metric:
        fpr, tpr, thresholds = metrics.roc_curve(y, y_predict, pos_label=2)
        metric_dict['auc'] = metrics.auc(fpr, tpr)
    if 'msle' in metric:
        metric_dict['msle'] = metrics.mean_squared_log_error(y, y_predict)
    if 'mse' in metric:
        metric_dict['mse'] = metrics.mean_squared_error(y, y_predict)
    if 'mae' in metric:
        metric_dict['mae'] = metrics.median_absolute_error(y, y_predict)
    return metric_dict
