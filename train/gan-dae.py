import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets

import model.GAN.GAN as gan

if __name__ == '__main__':
    # 设置部分超参数
    epoch = 10
    batch_size = 32
    # 加载训练数据
    mnist = datasets.MNIST(root='.\\data\\MNIST', download=True)
    mnist_data = mnist.data.data
    mnist_target = mnist.targets.data
    train_size = int(len(mnist_data) * 0.8)
    test_size = len(mnist_data) - train_size
    mnist_train_input, mnist_test_input = data.random_split(mnist_data, [train_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    mnist_train_target, mnist_test_target = data.random_split(mnist_target, [train_size, test_size],
                                                              generator=torch.Generator().manual_seed(42))
    mnist_train_input = data.DataLoader(mnist_train_input, batch_size=batch_size, shuffle=True, num_workers=2)
    mnist_test_input = data.DataLoader(mnist_test_input, shuffle=True, num_workers=2)
    # 定义模型和优化器及损失函数
    D_net = gan.Discriminator()
    G_net = gan.Generator()
    D_optimizer = optim.Adam(D_net.parameters(), lr=0.001, betas=(0.5, 0.999))
    G_optimizer = optim.Adam(G_net.parameters(), lr=0.001, betas=(0.5, 0.999))
    bce_loss = nn.BCELoss()
    for i in range(epoch):
        for x_real in mnist_train_input:
            real_x = x_real.view(-1, 28 * 28).to(torch.float32)
            rand_init = torch.randn(size=(batch_size, 32))

            with torch.autograd.set_detect_anomaly(True):
                fake_label = torch.zeros(size=(batch_size, 1))
                real_label = torch.ones(size=(batch_size, 1))
                fake_x = G_net(rand_init)

                z, d_predict_real = D_net(real_x)

                d_loss = bce_loss(d_predict_real, real_label) + bce_loss(D_net(fake_x.detach())[1], fake_label)

                D_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                D_optimizer.step()

                _, d_predict_fake = D_net(G_net(z.detach()))
                g_loss = bce_loss(d_predict_fake, real_label)

                G_optimizer.zero_grad()
                g_loss.backward()
                G_optimizer.step()

                print('epoch {}: g_loss={:.4f}, d_loss={:.4f}'.format(i, g_loss, d_loss))
                gan_params = {
                    'g_net_state_dict': G_net.state_dict(),
                    'd_net_state_dict': D_net.state_dict(),
                    'g_optimizer_state_dict': G_optimizer.state_dict(),
                    'd_optimizer_state_dict': G_optimizer.state_dict()
                }
    torch.save(gan_params, '../model_params/gan-2.pth')
