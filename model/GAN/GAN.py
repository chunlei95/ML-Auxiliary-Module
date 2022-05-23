import torch.nn as nn


class Discriminator(nn.Module):
    """判别器网络
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)


class Generator(nn.Module):
    """生成器网络

    """

    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.01),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.01),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),

            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
