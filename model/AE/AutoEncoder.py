import torch.nn as nn


class AutoEncoder(nn.Module):
    """自编码器模型

    :param encoder_blocks: 组成编码器模块的层
    :param decoder_blocks: 组成解码器模块的层
    """

    def __init__(self, encoder_blocks, decoder_blocks):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(encoder_blocks)
        self.decoder = Decoder(decoder_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    """编码器网络部分

    :param layers: 编码器使用的网络层序列
    """

    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layer_list = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_list(x)
        return x


class Decoder(nn.Module):
    """解码器网络部分

    :param layers: 解码器使用的网络层序列
    """

    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layer_list = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer_list(x)
        return x
