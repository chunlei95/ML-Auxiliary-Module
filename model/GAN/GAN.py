import torch.nn as nn


class Discriminator(nn.Module):
    """判别器网络

    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.BatchNorm1d(784),
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )
        self.linear = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.dis(x)
        return z, self.sigmoid(self.linear(z))


class Generator(nn.Module):
    """生成器网络

    """

    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
