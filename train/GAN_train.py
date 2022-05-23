import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
import model.GAN.GAN as gan
from torch.utils.tensorboard import SummaryWriter
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # 设置部分超参数
    continue_train = False
    epoch = 200
    batch_size = 64
    z_inputs = 64
    step = 0
    transform_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # 加载训练数据
    mnist = datasets.MNIST(root='..\\data\\mnist', transform=transform_list, download=True)
    mnist_train = data.DataLoader(mnist, batch_size=batch_size, shuffle=True, drop_last=True)
    # 定义模型和优化器及损失函数
    D_net = gan.Discriminator()
    G_net = gan.Generator()
    D_optimizer = optim.Adam(D_net.parameters(), lr=3e-4)
    G_optimizer = optim.Adam(G_net.parameters(), lr=3e-4)

    bce_loss = nn.BCELoss()
    fix_noise = torch.randn((batch_size, z_inputs))

    if continue_train:
        checkpoint_100_epoch = torch.load('../model_params/gan-mnist-100-result.pth')
        D_net.load_state_dict(checkpoint_100_epoch['d_net_state_dict'])
        G_net.load_state_dict(checkpoint_100_epoch['g_net_state_dict'])
        D_optimizer.load_state_dict(checkpoint_100_epoch['d_optimizer_state_dict'])
        G_optimizer.load_state_dict(checkpoint_100_epoch['g_optimizer_state_dict'])
        D_net.train()
        G_net.train()
        real_writer = SummaryWriter(f'../log_200/real')
        fake_writer = SummaryWriter(f'../log_200/fake')
    else:
        fake_writer = SummaryWriter(f'../log/fake')
        real_writer = SummaryWriter(f'../log/real')
        real_noise_writer = SummaryWriter(f'../log/noise')

    for i in range(epoch):
        for index, (x_real, _) in enumerate(mnist_train):
            real_x = x_real.view(-1, 28 * 28)
            full_rand_noise = torch.randn(real_x.shape)
            noise_x = torch.add(real_x, full_rand_noise)
            rand_noise = torch.randn(size=(batch_size, z_inputs))

            fake_x = G_net(rand_noise)

            fake_label = torch.zeros(size=(batch_size, 1)).squeeze()
            real_label = torch.ones(size=(batch_size, 1)).squeeze()

            d_predict_real = D_net(noise_x).squeeze()
            d_predict_fake = D_net(fake_x).squeeze()

            d_loss = (bce_loss(d_predict_real, real_label) + bce_loss(d_predict_fake, fake_label)) / 2

            D_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            D_optimizer.step()

            # z = Variable(G_net(rand_noise))
            d_predict_fake = D_net(fake_x).squeeze()
            g_loss = bce_loss(d_predict_fake, real_label)

            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            if index == 0:
                print('epoch {}: batch {}/{} g_loss={:.4f}, d_loss={:.4f}'.format(i, index, len(mnist_train),
                                                                                  g_loss, d_loss))
                with torch.no_grad():
                    fake = G_net(fix_noise).reshape(-1, 1, 28, 28)
                    real = real_x.reshape(-1, 1, 28, 28)
                    noise = noise_x.reshape(-1, 1, 28, 28)

                    fake_image_grid = torchvision.utils.make_grid(fake, normalize=True)
                    real_image_grid = torchvision.utils.make_grid(real, normalize=True)
                    noise_image_grid = torchvision.utils.make_grid(noise, normalize=True)

                    fake_image_name = 'fake_image_200' if continue_train else 'fake_image'
                    real_image_name = 'real_image_200' if continue_train else 'real_image'
                    noise_image_name = 'noise_image_200' if continue_train else 'noise_image'

                    fake_writer.add_image(fake_image_name, fake_image_grid, global_step=step)
                    real_writer.add_image(real_image_name, real_image_grid, global_step=step)
                    real_writer.add_image(noise_image_name, noise_image_grid, global_step=step)

                    step += 1

    model_train_params = {
        'g_net_state_dict': G_net.state_dict(),
        'd_net_state_dict': D_net.state_dict(),
        'g_optimizer_state_dict': G_optimizer.state_dict(),
        'd_optimizer_state_dict': G_optimizer.state_dict(),
    }
    path = '../model_params/gan-noise-result.pth' if continue_train else '../model_params/gan-result.pth'
    torch.save(model_train_params, path)
