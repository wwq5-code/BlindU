import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from net import *
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import sys


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))



class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


############## dcgan ##############
class Generator(nn.Module):
    def __init__(self, resnet, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()

        self.resnet = resnet
        self.linear = nn.Linear(1402, 224)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(4*3*32*32, 3*32*32)
        self.bn1 = nn.BatchNorm1d(3*32*32)
        self.bn2 = nn.BatchNorm2d(3)
        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, noise, grad):
        grad_linear = self.linear(grad)
        concatenated_tensor = torch.cat((noise, grad_linear), dim=1)
        x = concatenated_tensor.view(-1, 1, 32, 32)
        x = self.resnet(x)
        # x = self.fc1(x)
        # x = self.bn1(x)
        # x = nn.ReLU()(x)
        x = x.view(-1,3,32,32)
        # x = self.bn2(x)
        output = nn.Tanh()(x)
        # output = self.main(x)
        # print(output.shape)
        # output = output.view(-1,3,32,32)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_embeddings=1402, embedding_dim=nc)

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, grad):
        # deal the x to [64,3,16,16]
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        x = max_pool(x)
        # deal the grad to [64,3,16,16]
        linear_layer = nn.Linear(1402, 3 * 16 * 16).to(device)
        grad = linear_layer(grad).view(grad.size(0), 3, 16, 16).to(device)

        # firs half
        x1 = torch.cat((x, grad), dim=2)
        x2 = torch.cat((x, grad), dim=2)
        x = torch.cat((x1, x2), dim=3)
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


@torch.no_grad()
def test_loader_eva(model, data_loader, device):
    model.eval()
    for i, (data) in enumerate(data_loader):
        data_z, label, grad, _ = data
        batch_size_for_t = len(data_z)
        data_z, label, grad = Variable(data_z).to(device), Variable(label).to(device), Variable(grad).to(device)

        z = torch.randn(batch_size_for_t, 800, device=device)
        x_hat = model(z, grad)
        show_cifar(data_z)
        show_cifar(x_hat)
        break



def show_cifar(x):
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)

    x = x.view(x.size(0), 3, 32, 32)
    # print(x)
    grid = torchvision.utils.make_grid(x, nrow=4, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()


FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

device = 'cuda:0'
batch_size = 64
batch_size_for_t = 32
lr = 0.0005
betas = (0.5, 0.999)
epochs = 250

# ll_dataset = torch.load('cifar_2k_1402_withmodel.list',map_location=device)

# Define the split lengths
train_length = int(0.995 * len(ll_dataset))
test_length = len(ll_dataset) - train_length

# Split the dataset into train and test sets
trainset, testset = random_split(ll_dataset, [train_length, test_length])


loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,  # 乱序
)

test_loader = DataLoader(
    testset,
    batch_size=batch_size_for_t,
    shuffle=True,  # 乱序
)


# block_features = [64] * 10
# resnet18 = ResNet(1, block_features, 3, headless=True)

# block_features = [64] * 2 + [128] * 3 + [256] * 4
# resnet18 = ResNet(1, block_features, 3, headless=True)

attacker = Generator(resnet=resnet18(1,3*32*32)).to(device)
optimizer_G = torch.optim.Adam(attacker.parameters(), lr=lr, betas=betas)
discriminator = Discriminator().to(device)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
attacker.apply(weights_init)
discriminator.apply(weights_init)

for epoch in range(epochs):
    attacker.train()
    discriminator.train()
    for i, (data) in enumerate(loader):
        data_z, label, grad, _ = data
        batch_size = len(data_z)
        data_z, label, grad = Variable(data_z).to(device), Variable(label).to(device), Variable(grad).to(device)
        valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False).to(device)
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, 800, device=device)
        x_hat = attacker(z, grad)
        validity_real = discriminator(data_z, grad)
        d_real_loss = torch.nn.BCELoss()(validity_real, valid)
        d_real_loss.backward()
        validity_fake = discriminator(x_hat.detach(), grad)
        d_fake_loss = torch.nn.BCELoss()(validity_fake, fake)
        d_fake_loss.backward()
        d_loss = (d_real_loss + d_fake_loss) / 1
        optimizer_D.step()

        # train attacker
        optimizer_G.zero_grad()
        # x_hat = attacker(grad)
        grad_loss = 0
        validity = discriminator(x_hat, grad)
        mseloss = torch.nn.MSELoss(reduction='sum')(x_hat, data_z)/1000
        g_loss = torch.nn.BCELoss()(validity, valid) / 30.0 + mseloss #* 30.0
        g_loss.backward()
        optimizer_G.step()
    if epoch % 1 == 0:
        print('[%d,%d] mseloss:%.3f g_loss:%.3f d_loss:%.3f' % (
        epoch + 1, epochs, mseloss.item(), g_loss.item(), d_loss.item()))
        # show_cifar(data_z)
        # show_cifar(x_hat)
        a = test_loader_eva(attacker,  test_loader, device)








def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cost(pred, target, device):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1)).to(device)


class LeNet(torch.nn.Module):
    def __init__(self, channel=1, hideen=588, num_classes=10):
        super(LeNet, self).__init__()
        act = torch.nn.ReLU
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(channel, 4, kernel_size=3),
            act(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 8, kernel_size=3),
            act(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 32, kernel_size=3),
            act(),
            torch.nn.MaxPool2d(2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
