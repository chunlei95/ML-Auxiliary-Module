import torch.nn as nn
import torch.optim as optim
import utils.train as train
from model.AE.AutoEncoder import AutoEncoder


def get_data():
    pass


if __name__ == '__main__':
    encoder_blocks = []
    decoder_blocks = []
    ae = AutoEncoder(encoder_blocks, decoder_blocks)
    criteria = nn.MSELoss()
    train_data = get_data()
    optimizer = optim.Adam(ae.parameters(), lr=0.001, weight_decay=0.05)
    train.trainer(train_data, ae, optimizer, criteria, {})
