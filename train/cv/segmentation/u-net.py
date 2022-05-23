import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.cv.segmentation.UNet.UNet import UNet

params = {
    'epoch': 100,
    'lr': 1e-4,
    'betas': (0.5, 0.999)
}


def load_data():
    cityscapes_semantic = datasets.VOCSegmentation("../../../data/VOCSegmentation", download=True)
    images, semantics = cityscapes_semantic[0]
    train_datas = []
    test_datas = []
    train_labels = []
    test_labels = []
    trains = DataLoader(train_datas, batch_size=32, shuffle=True, num_workers=4)
    return trains


def im_processing(train_datas):
    """对图像做预处理

    :param train_datas: 训练集
    :return: 预处理之后的图像数据集
    """
    return train_datas


def trainer(model, optimizer, loss, epoch, device=None):
    train_datas, train_labels, test_data, test_labels = load_data()
    train_datas = im_processing(train_datas)
    if device is not None:
        model.to(device)
        optimizer.to(device)
        loss.to(device)
    epoch_index_list = []
    loss_change_list = []
    for i in range(epoch):
        total_loss = 0.0
        total_count = 0
        for image, labels in zip(train_datas, train_labels):
            # todo 图像输入之前进行预处理

            # 图像输入之间作标准化处理,image.shape[1]是获取输入图像的通道数
            batch_norm = nn.BatchNorm2d(image.shape[1])
            segment_mask = model(batch_norm(image))
            loss_value = loss(segment_mask, labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            total_loss = total_loss + loss_value
            total_count = total_count + len(labels)
            # 每隔10个epoch，记录损失值，用于绘制损失变化 ****************
        if i % 10 == 0:
            epoch_index_list.append(i)
            loss_change_list.append(total_loss / total_count)
        # ******************************************************
    torch.save(model.state_dicts(), '../../../model_params/u_net_voc_seg.pth')


def test(model, test_data, batch_size):
    pass


if __name__ == '__main__':
    current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = UNet().to(current_device)
    cross_entropy_loss = nn.CrossEntropyLoss().to(current_device)
    optimizer_adam = optim.Adam(unet.parameters(), lr=params.get("lr"), betas=params.get("betas"))
    load_data()
