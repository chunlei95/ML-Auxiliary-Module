import torch
import matplotlib.pyplot as plt

from model.GAN.GAN import Generator

g_net = Generator()
unet_train_params = torch.load('../model_params/gan-mnist-100-result.pth')
g_net_params = unet_train_params['g_net_state_dict']
g_net.load_state_dict(g_net_params)
g_net.eval()
result = g_net(torch.randn(size=(64, 64)))
images = result[1]
images = images.view((28, 28))
figure, axes = plt.subplots()
row = 0
column = 0
# for i in range(len(images)):
#     # row = i % 8
#     # column = (i + 4) % 4
#     axes[row][column].imshow(images[i].detach().numpy())
axes.imshow(images.detach().numpy())
plt.show()
