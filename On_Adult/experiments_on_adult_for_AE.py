import sys

sys.argv = ['']
del sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import argparse
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import copy
import math
import random
import time


class LinearModel(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)
    args = parser.parse_args()
    return args


class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder


    def explain(self, x, mode='topk'):
        """Returns the relevance scores
        """
        double_logits_z = self.encoder(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'with64QAM_distribution':
            B, double_dimZ = double_logits_z.shape
            #double_dimZ_after_QAM = process_64QAM(double_logits_z)
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 2))  # (B,   10) binary for adult dataset
            return logits_z, logits_y, mu, logvar
        elif mode == '64QAM_distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #print(logits_z)
            # convert to bit
            logits_z_sig = torch.sigmoid(logits_z)
            logits_z_sig = (logits_z_sig > 0.5).float()

            # Reshape to (batch_size, symbol_count, bits_per_symbol)
            logits_z_sig = logits_z_sig.view(logits_z_sig.shape[0], -1, 6).to('cuda')

            # QAM modulation
            logits_z_arr = self.qam_modulation(logits_z_sig)

            # change QAM modulated complex number to tensor
            real_part = torch.real(logits_z_arr)
            imag_part = torch.imag(logits_z_arr)

            input_tensor = torch.cat([real_part.unsqueeze(1), imag_part.unsqueeze(1)], dim=1).cuda()

            logits_y = self.approximator(input_tensor)  # (B , 10)
            logits_y = logits_y.reshape((B, 2))  # (B,   10) 2 for adult
            return logits_z, logits_y, mu, logvar

        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'with64QAM_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='with64QAM_distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            #logits_z_after_QAM = process_64QAM(logits_z)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # VAE is not related to labels
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # logits_y = self.approximator(logits_z)  # (B , 10)
            # logits_y = logits_y.reshape((B, 10))  # (B,   10)
            # in general, a larger K factor in Rician noise could be considered "better" from a signal quality perspective
            #logits_z_with_awgn = add_rician_noise(logits_z, torch.tensor(args.SNR).cuda(), K=5)  # add_awgn_noise ,  add_rayleigh_noise,
            #logits_z_with_awgn = add_rician_noise(logits_z, torch.tensor(args.SNR).cuda(), K=torch.tensor(2).cuda())
            x_hat = self.reconstruction(logits_z)
            return logits_z, x_hat, mu, logvar
        elif mode == '64QAM_VAE':
            logits_z, mu, logvar = self.explain(x, mode='with64QAM_distribution')  # (B, C, H, W), (B, C* h* w)
            #print(logits_z)
            #logits_z_after_QAM = process_64QAM(logits_z)
            #print("input_tensor",input_tensor.shape)
            x_hat = self.reconstruction(logits_z)
            return logits_z, x_hat, mu, logvar
        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y
        elif mode == 'forgetting_from_Z':
            #logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_z = x
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 2))  # (B,   10) 2 for adult
            #x_hat = self.forget(logits_z)
            return logits_z, logits_y# , x_hat #, mu, logvar

    def reconstruction(self, logits_z):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.decoder(logits_z)
        return torch.sigmoid(output_x)

    def cifar_recon(self, logits_z):
        # B, c, h, w = logits_z.shape
        # logits_z=logits_z.reshape((B,-1))
        output_x = self.reconstructor(logits_z)
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


def init_vib(args):
    if args.dataset == 'MNIST':
        approximator = LinearModel(n_feature=args.dimZ)
        decoder = LinearModel(n_feature=args.dimZ, n_output=28 * 28)
        #encoder = QAM64Encoder(qam_modulation=qam_modulation, n_feature=28 * 28, n_output=args.dimZ * 2)  # resnet18(1, 49*2) #
        encoder = LinearModel(n_feature=28 * 28, n_output=args.dimZ * 2)  # 64QAM needs 6 bits
        lr = args.lr

    elif args.dataset == 'CIFAR10':
        # approximator = resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        approximator = LinearModel(n_feature=args.dimZ)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr
    elif args.dataset =='Adult':
        approximator = LinearModel(n_feature=args.dimZ, n_output=2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=14)
        encoder = LinearModel(n_feature=14, n_output=args.dimZ*2)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr


def vib_train(dataset, model, loss_fn, reconstruction_function, args, epoch, mu_list, sigma_list, train_loader, train_bs_begin, train_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)

        logits_z, logits_y, mu, logvar = model(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q = loss_fn(logits_y, y)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        #x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        #BCE = reconstruction_function(x_hat, x)  # mse loss for vae

        if train_type == 'NIPS':
            loss =  H_p_q #args.beta * KLD_mean +
        elif train_type == 'VIB':
            loss = args.beta * KLD_mean + H_p_q
        #loss = args.beta * KLD_mean + BCE  # / (args.batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'loss': loss.item(),
            #'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }
        train_bs_begin = train_bs_begin + 1
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            # print(x)
            # print(y)
            # x_hat_cpu = x_hat.cpu().data
            # x_hat_cpu = x_hat_cpu.clamp(0, 1)
            # x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
            # grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()
    return model, mu_list, sigma_list, train_bs_begin


def vib_infer_train(dataset, model, loss_fn, infer_classifier, infer_classifier_of_grad, args, epoch, mu_list, sigma_list, train_loader,train_bs_begin):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_inf = torch.optim.Adam(infer_classifier.parameters(), lr=args.lr)
    optimizer_inf_grad = torch.optim.Adam(infer_classifier_of_grad.parameters(), lr=args.lr)
    for step, (x, y, inf_label) in enumerate(dataset):
        x, y, inf_label= x.to(args.device), y.to(args.device), inf_label.to(args.device) # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)

        logits_z, logits_y, mu, logvar = model(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q = loss_fn(logits_y, y)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        #x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        #BCE = reconstruction_function(x_hat, x)  # mse loss for vae

        loss = args.beta * KLD_mean + H_p_q #args.beta * KLD_mean +
        #loss = args.beta * KLD_mean + BCE  # / (args.batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()


        logits_y_inf = infer_classifier(logits_z.detach())  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q_inf = loss_fn(logits_y_inf, inf_label)

        loss_inf = H_p_q_inf
        optimizer_inf.zero_grad()
        loss_inf.backward()
        torch.nn.utils.clip_grad_norm_(infer_classifier.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer_inf.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == 'encoder.fc2.weight':
                    fc2_grad = param.grad
                    fc2_grad = fc2_grad.reshape((1,-1))
                    # print(fc2_grad.shape)
                # if param.grad ==None: continue
                # print(name, param.grad.shape)

        # break

        logits_y_inf_grad = infer_classifier_of_grad(fc2_grad.detach())  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        #used to train infer model from grad
        # H_p_q_inf_grad = loss_fn(logits_y_inf_grad, inf_label)
        #
        # loss_inf_grad = H_p_q_inf_grad
        # optimizer_inf_grad.zero_grad()
        # loss_inf_grad.backward()
        # torch.nn.utils.clip_grad_norm_(infer_classifier.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        # optimizer_inf_grad.step()

        acc_inf_grad = (logits_y_inf_grad.argmax(dim=1) == inf_label).float().mean().item()
        acc_inf = (logits_y_inf.argmax(dim=1) == inf_label).float().mean().item()
        acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'acc_inf': acc_inf,
            'loss': loss.item(),
            #'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }
        train_bs_begin=train_bs_begin+1
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            # print(x)
            # print(y)
            # x_hat_cpu = x_hat.cpu().data
            # x_hat_cpu = x_hat_cpu.clamp(0, 1)
            # x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
            # grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

    return model, mu_list, sigma_list, infer_classifier, infer_classifier_of_grad, train_bs_begin

def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def calculate_MI(X, Z, Z_size, M, M_opt, args, ma_rate=0.001):
    '''
    we use Mine to calculate the mutual information between two layers of networks.
    :param G:
    :param M:
    :param ma_rate:
    :return:
    '''

    z_bar = torch.randn((args.batch_size, Z_size)).to(args.device)

    et = torch.mean(torch.exp(M(z_bar, X)))

    if M.ma_et is None:
        M.ma_et = et.detach().item()

    M.ma_et += ma_rate * (et.detach().item() - M.ma_et)

    #z = torch.narrow(z, dim=1, start=0, length=3)  # slice for MI
    mutual_information = torch.mean(M(Z, X)) \
                         - torch.log(et) * et.detach() / M.ma_et

    loss = - mutual_information

    M_opt.zero_grad()
    loss.backward()
    M_opt.step()

    return mutual_information.item()


class Mine1(nn.Module):

    def __init__(self, noise_size=49, sample_size=28*28, output_size=1, hidden_size=128):
        super(Mine1, self).__init__()
        self.fc1_noise = nn.Linear(noise_size, hidden_size, bias=False)
        self.fc1_sample = nn.Linear(sample_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.ma_et = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, noise, sample):
        x_noise = self.fc1_noise(noise)
        x_sample = self.fc1_sample(sample)
        x = F.leaky_relu(x_noise + x_sample + self.fc1_bias, negative_slope=2e-1)
        x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
        x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
        return x



@torch.no_grad()
def eva_vib(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()
    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        logits_z, logits_y, mu, logvar = vib(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}')
    return acc

@torch.no_grad()
def eva_vib_inf(vib, classifier, dataloader_erase, args, name='infer', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()
    num_total = 0
    num_correct = 0
    for batch_idx, (x, y, y_inf) in enumerate(dataloader_erase):
        x, y, y_inf = x.to(args.device), y.to(args.device), y_inf.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        logits_z, logits_y, mu, logvar = vib(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        logits_y_inf = classifier(logits_z)
        num_correct += (logits_y_inf.argmax(dim=1) == y_inf).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}')
    return acc


def eva_vib_inf_grad(vib, classifier, dataloader_erase, args, name='infer', epoch=999):
    # first, generate x_hat from trained vae
    # vib.eval()
    num_total = 0
    num_correct = 0
    optimizer = torch.optim.Adam(vib.parameters(), lr=args.lr)

    for batch_idx, (x, y, y_inf) in enumerate(dataloader_erase):
        x, y, y_inf = x.to(args.device), y.to(args.device), y_inf.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)

        logits_z, logits_y, mu, logvar = vib(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
        if y.ndim == 2:
            y = y.argmax(dim=1)

        H_p_q = loss_fn(logits_y, y)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        #x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        #BCE = reconstruction_function(x_hat, x)  # mse loss for vae

        loss =  H_p_q #args.beta * KLD_mean +
        #loss = args.beta * KLD_mean + BCE  # / (args.batch_size * 28 * 28)
        optimizer.zero_grad()
        loss.backward()

        for name, param in vib.named_parameters():
            if param.requires_grad:
                if name == 'encoder.fc2.weight':
                    fc2_grad = param.grad
                    fc2_grad = fc2_grad.reshape((1,-1))
                # if param.grad ==None: continue
                # print(name, param.grad.shape)

        # break

        logits_y_inf_grad = classifier(fc2_grad.detach())  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        # logits_y_inf = classifier(logits_z)
        num_correct += (logits_y_inf_grad.argmax(dim=1) == y_inf).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}')
    return acc

@torch.no_grad()
def test_accuracy(model, data_loader, args, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        if args.dataset == 'MNIST':
            x = x.view(x.size(0), -1)
        out = model(x, mode='test')
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    # print(f'{name} accuracy: {acc:.4f}')
    return acc

def unlearning_frkl_compressed(vibi_f_frkl, optimizer_frkl, vibi, epoch_test_acc, dataloader_erase, dataloader_remain, loss_fn,
                    reconstructor, reconstruction_function, test_loader, train_loader, train_type):


    acc_test = []
    backdoor_acc_list = []

    print(len(dataloader_erase.dataset))
    train_bs = 0
    temp_acc = []
    temp_back = []

    for epoch in range(args.num_epochs+10): #args.num_epochs
        vibi_f_frkl.train()
        step_start = epoch * len(dataloader_erase)
        index = 0
        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
            vibi_f_frkl.train()
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST' or args.dataset=='Adult':
                x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST' or args.dataset=='Adult':
                x2 = x2.view(x2.size(0), -1)

            # y=y+1
            # print('y',y)
            # print('y2', y2)
            #logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            #logits_z_e2, logits_y_e2, x_hat_e2, mu_e2, logvar_e2 = vibi_f_frkl(x2, mode='forgetting')
            logits_z_f, logits_y_f, mu_f, logvar_f = vibi(x, mode='distribution')
            #logits_z_f2, logits_y_f2, x_hat_f2, mu_f2, logvar_f2 = vibi(x2, mode='forgetting')


            #here, the logvar is simga.pow(2)
            if train_type == 'NIPSU':
                logits_z_e, logits_y_e, mu_e, logvar_e = vibi_f_frkl(x, mode='distribution')
                KLD_mean = 0.5 * torch.mean(logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()
            else:
                logits_z_e, logits_y_e  = vibi_f_frkl(logits_z_f, mode='forgetting_from_Z')
            #logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            #logits_z_e2, logits_y_e2, x_hat_e2  = vibi_f_frkl(logits_z_f2, mode='forgetting_from_Z')

            #x2 = torch.rand(len(x2), 28*28).to(args.device)

            logits_z_e2, logits_y_e2, mu_e2, logvar_e2  = vibi_f_frkl(x2, mode='distribution')

            #two distributions, p(Z_e|X_e) and p(Z|X), apart from the q(Z)
            # logits_y_e = torch.softmax(logits_y_e, dim=1)
            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)

            H_p_q = loss_fn(logits_y_e, y)

            H_p_q2 = loss_fn(logits_y_e2, y2)

            # print()
            # print()
            if logvar_e2.shape != logvar_f.shape:
                continue
            # kl[f||e], where f is original, e is new trained based on remaining dataset
            KLD_mean_z_and_z_e = 0.5 * torch.mean(
                logvar_e2 - logvar_f + (torch.exp(logvar_f) + (mu_f - mu_e2).pow(2)) / torch.exp(logvar_e2) - 1).to(args.device)


            KLD_element_f = mu_f.pow(2).add_(logvar_f.exp()).mul_(-1).add_(1).add_(logvar_f).to(args.device)
            KLD_mean_f = torch.mean(KLD_element_f).mul_(-0.5).to(args.device)


            # KLD_element_e = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e).to(args.device)
            # KLD_mean_e = torch.mean(KLD_element_e).mul_(-0.5).to(args.device)


            KLD_element2 = mu_e2.pow(2).add_(logvar_e2.exp()).mul_(-1).add_(1).add_(logvar_e2).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            if KLD_mean_z_and_z_e.item() < KLD_mean_f.item():
                unlearn_rep = 1/(KLD_mean_z_and_z_e+0.001)
            else:
                unlearn_rep = 0*1/(KLD_mean_z_and_z_e+0.001)


            rho = torch.div(torch.mul(logits_z_e2 - mu_e2, logits_z_f - mu_f), torch.mul(logvar_e2, logvar_f) )
            mean_rho = torch.mean(rho)

            mutual_info_ratio = 0.00001
            if mean_rho.pow(2) >=1:
                mean_rho= torch.tensor([0.99])
                mutual_info_ratio=0
            mutual_info_z_and_ze = torch.mean(1/2 * torch.log(1 - mean_rho.pow(2)) ).to(args.device)
            '''
            KLD_element2 = mu_e2.pow(2).add_(logvar_e2.exp()).mul_(-1).add_(1).add_(logvar_e2).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            # kl[e||f], where e is q(Z_e|X_e), if is p(Z|X)
            KLD = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KLD_mean = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KL_z_r = (torch.exp(logits_z_e_log_softmax) * logits_z_e_log_softmax).sum(dim=1).mean() + math.log(
                logits_z_e_log_softmax.shape[1])
            '''

            M = Mine1(noise_size=args.dimZ, sample_size=args.dimZ)
            M.to(args.device)
            M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)

            B, Z_size = logits_z_e.shape
            for i in range(args.mi_epoch):
                mi = calculate_MI(logits_z_e2.detach(), logits_z_e.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
                if mi < 0:
                    i=i-1

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

            # x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
            # x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(x)
            #BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            #e_log_p = torch.exp(BCE / (args.batch_size * 28 * 28))  # = 1/p
            e_log_py = torch.exp(H_p_q)
            log_z = torch.mean(logits_z_e.log_softmax(dim=1))
            log_y = torch.mean(logits_y_e.log_softmax(dim=1))
            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            kl_f_e = kl_loss(F.log_softmax(logits_y_e, dim=1), F.log_softmax(logits_y_f, dim=1))
            # loss = args.beta * KLD_mean + H_p_q - BCE / (args.batch_size * 28 * 28) - log_z / e_log_p

            # loss = KLD_mean - BCE + args.unlearn_learning_rate * (
            #             kl_f_e - H_p_q)  # #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2

            #print(KLD_mean.item(), BCE.item(), kl_f_e.item(), H_p_q.item(), log_z.item(), log_y.item(), H_p_q2.item())

            #1/KLD_mean_z_and_z_e, KLD_mean_f.item(), 0.0000 * mutual_info_z_and_ze.item() +




            # original calculate, it is better for forgetting
            unlearning_item = args.mi_rate * mi - args.unlearn_learning_rate * H_p_q.item()

            learning_item = args.self_sharing_rate * (args.beta * KLD_mean2.item() + H_p_q2.item() ) #

            total = unlearning_item + learning_item + 20 # expected to equal to 0
            if unlearning_item <= - 20:# have approixmate to the retrained distribution and no need to unlearn
                unl_rate = 0.0
            else:
                unl_rate = (unlearning_item + 20) / total

            self_s_rate = 1 - unl_rate

            # Calculate the L2-norm
            l2_norm_unl = torch.norm(args.mi_rate * mi - args.unlearn_learning_rate * H_p_q, p=2)

            l2_norm_ss = torch.norm(args.self_sharing_rate * (args.beta * KLD_mean2 + H_p_q2 ) , p=2)

            total_u_s = l2_norm_unl + l2_norm_ss
            unl_rate = l2_norm_unl / total_u_s
            self_s_rate = l2_norm_ss / total_u_s


            ''' purpose is to make the unlearning item =0, and the learning item =0 '''

            if train_type == 'VIBU':
                #loss = args.kld_r * KLD_mean - args.unlearn_bce_r * BCE + args.unlearn_ykl_r * kl_f_e - args.unlearn_learning_rate * H_p_q - args.reverse_rate * (log_z + log_y)
                #loss = args.beta * KLD_mean2 + mi - args.unlearn_learning_rate * H_p_q
                loss = args.mi_rate * mi - args.unlearn_learning_rate * H_p_q # + 0.1 * args.unlearn_learning_rate * kl_f_e
            elif train_type == 'VIBU-SS':
                # loss = (args.kld_r * KLD_mean - args.unlearn_bce_r * BCE + args.unlearn_ykl_r * kl_f_e -  args.unlearn_learning_rate * H_p_q - args.reverse_rate * (log_z + log_y) ) * unl_rate + self_s_rate * args.self_sharing_rate * (
                #  args.beta * KLD_mean2 +    KLD_mean_f +,   0.0000 * mutual_info_z_and_ze +,            args.beta * KLD_mean2 + H_p_q2)  # args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2

                loss = (args.mi_rate * mi - args.unlearn_learning_rate * H_p_q) * unl_rate + self_s_rate * args.self_sharing_rate * (args.beta * KLD_mean2 + H_p_q2)  # args.beta * KLD_mean2 + args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2
                #loss = ( - args.unlearn_learning_rate * H_p_q) * unl_rate + self_s_rate * args.self_sharing_rate * (H_p_q2)  # args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2
            elif train_type == 'NIPSU':
                loss = args.beta * KLD_mean*0.0 + args.unl_r_for_bayesian * (- H_p_q)
                #- args.reverse_rate * (log_z + log_y) args.kld_r * KLD_mean + - args.reverse_rate * (log_z + log_y)
                #loss =  KLD_mean_e*10 - args.unlearn_learning_rate * H_p_q - args.reverse_rate * (log_z + log_y)
                #loss = args.kld_r * KLD_mean + args.unl_r_for_bayesian * (- H_p_q) - args.reverse_rate * (log_z + log_y)


            optimizer_frkl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vibi_f_frkl.parameters(), args.max_norm, norm_type=2.0, error_if_nonfinite=False)
            optimizer_frkl.step()
            acc_back = (logits_y_e.argmax(dim=1) == y).float().mean().item()
            acc = (logits_y_e2.argmax(dim=1) == y2).float().mean().item()

            #acc_back = test_accuracy(vibi_f_frkl, dataloader_erase, args, name='vibi valid top1')
            temp_acc.append(acc)
            temp_back.append(acc_back)
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()

            # B, Z_size = logits_z_e2.shape
            # mi = calculate_MI(x.detach(), logits_z_e2.detach(), Z_size, M, M_opt, args, ma_rate=0.001)

            metrics = {
                'unlearning_item': unlearning_item,
                'learning_item': learning_item,
                'l2_unl_norm': l2_norm_unl.item(),
                'l2_s_norm': l2_norm_ss.item(),
                'acc': acc,
                'loss': loss.item(),
                #'BCE': BCE.item(),
                'H(p,q)': H_p_q.item(),
                'kl_f_e': kl_f_e.item(),
                'H_p_q2': H_p_q2.item(),
                'mutual_info_z_and_ze': mutual_info_z_and_ze.item(),
                # 'mu_e': torch.mean(mu_e).item(),
                # 'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                'KLD_mean_z_and_z_e': KLD_mean_z_and_z_e.item(),
                'torch.mean( rho )':torch.mean( rho ).item(),
                #'e_log_p': e_log_p.item(),
                'log_z': log_z.item(),
                'KLD_mean_f': KLD_mean_f.item(),
                'KLD_mean_e2':KLD_mean2.item(),
                'mutual_info':mi,
            }

            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu_e).item())
            #     sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if index % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{index % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            index = index + 1
            train_bs = train_bs+1
            if acc_back < 0.02: #and train_type =='NIPSU'
                break

        vibi_f_frkl.eval()
        valid_acc_old = 0.8
        valid_acc = test_accuracy(vibi_f_frkl, test_loader, args, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()

        print("test_acc", valid_acc)
        epoch_test_acc.append(valid_acc)
        print("epoch: ", epoch)
        # valid_acc_old = valid_acc
        # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        print("test_acc", valid_acc)
        acc_test.append(valid_acc)
        backdoor_acc = test_accuracy(vibi_f_frkl, dataloader_erase, args, name='vibi valid top1')
        backdoor_acc_list.append(backdoor_acc)
        print("backdoor_acc", backdoor_acc_list)
        print("acc_test: ", acc_test)
        if backdoor_acc < 0.1:
            print()
            print("end unlearn, train_bs", train_bs)
            break

    print("end unlearn, train_bs", train_bs)
    print("temp_acc", temp_acc)
    print("temp_back", temp_back)

    return vibi_f_frkl, optimizer_frkl, epoch_test_acc, valid_acc, backdoor_acc, train_bs


'''we should prepare the compressed Z, and then use Z to be unlearned. 
not directly unlearning as normal'''
def unlearning_frkl_train(vibi, dataloader_erase, dataloader_remain, loss_fn, reconstructor, reconstruction_function,
                          test_loader, train_loader, train_type='VIBU'):
    vibi_f_frkl, lr = init_vib(args)
    vibi_f_frkl.to(args.device)
    vibi_f_frkl.load_state_dict(vibi.state_dict())
    optimizer_frkl = torch.optim.Adam(vibi_f_frkl.parameters(), lr=lr)

    init_epoch = 0
    print("unlearning")

    if args.dataset=="MNIST":
        reconstructor_for_unlearning = LinearModel(n_feature=49, n_output=28 * 28)
        reconstructor_for_unlearning = reconstructor_for_unlearning.to(args.device)
        optimizer_recon_for_un = torch.optim.Adam(reconstructor_for_unlearning.parameters(), lr=lr)
    elif args.dataset=="CIFAR10":
        reconstructor_for_unlearning = resnet18(3, 3 * 32 * 32)
        reconstructor_for_unlearning = reconstructor_for_unlearning.to(args.device)
        optimizer_recon_for_un = torch.optim.Adam(reconstructor_for_unlearning.parameters(), lr=lr)


    if init_epoch == 0 or args.resume_training:

        # print('Unlearning VIBI KLD')
        # print(f'{args.explainer_type:>10} explainer params:\t{num_params(vibi_f_frkl.explainer) / 1000:.2f} K')
        # print(
        #     f'{type(vibi_f_frkl.approximator).__name__:>10} approximator params:\t{num_params(vibi_f_frkl.approximator) / 1000:.2f} K')
        # print(
        #     f'{type(vibi_f_frkl.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_f_frkl.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        epoch_test_acc = []
        mu_list = []
        sigma_list = []
        ''' we should fix the original model to fix the Z
        here, vibi_f_frkl is the unlearning model and vibi is the original model 
        so we can used the original model to create the fixed Z
        '''
        vibi_f_frkl, optimizer_frkl, epoch_test_acc, valid_acc, backdoor_acc, train_bs = unlearning_frkl_compressed(vibi_f_frkl, optimizer_frkl, vibi,
                                                                      epoch_test_acc, dataloader_erase,
                                                                      dataloader_remain, loss_fn,
                                                                      reconstructor, reconstruction_function,
                                                                      test_loader, train_loader, train_type)

        # final_round_mse = []
        # for epoch in range(init_epoch, init_epoch + args.num_epochs):
        #     vibi.train()
        #     step_start = epoch * len(dataloader_erase)
        #     for step, (x, y) in enumerate(dataloader_erase, start=step_start):
        #         x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        #         x = x.view(x.size(0), -1)
        #         logits_z, logits_y, x_hat_e, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        #
        #         x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
        #         x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
        #         x = x.view(x.size(0), -1)
        #         # x = torch.sigmoid(torch.relu(x))
        #         BCE = reconstruction_function(x_hat_e, x)  # mse loss
        #         loss = BCE
        #
        #         optimizer_recon_for_un.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        #         optimizer_recon_for_un.step()
        #         if epoch == args.num_epochs - 1:
        #             final_round_mse.append(BCE.item())
        #         if step % len(train_loader) % 600 == 0:
        #             print("loss", loss.item(), 'BCE', BCE.item())
        #
        # print("final epoch mse", np.mean(final_round_mse))
        #
        # for step, (x, y) in enumerate(test_loader):
        #     x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        #     x = x.view(x.size(0), -1)
        #     logits_z, logits_y, x_hat_e, mu, logvar = vibi_f_frkl(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        #     x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
        #     x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
        #     x = x.view(x.size(0), -1)
        #     break
        #
        # print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))
        # print("frkld epoch_test_acc", epoch_test_acc)
        # x_hat_e_cpu = x_hat_e.cpu().data
        # x_hat_e_cpu = x_hat_e_cpu.clamp(0, 1)
        # x_hat_e_cpu = x_hat_e_cpu.view(x_hat_e_cpu.size(0), 1, 28, 28)
        # grid = torchvision.utils.make_grid(x_hat_e_cpu, nrow=4, cmap="gray")
        # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        # plt.show()
        #
        # x_cpu = x.cpu().data
        # x_cpu = x_cpu.clamp(0, 1)
        # x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
        # grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
        # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        # plt.show()
    return vibi_f_frkl, optimizer_frkl, valid_acc,  backdoor_acc, train_bs


def scoring_function(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Check if the matrix has a constant value to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(matrix)

    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix


def dp_sampling(matrix, epsilon, sample_size, replacement):
    scores = scoring_function(matrix.numpy())
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities = probabilities / probabilities.sum()
    # probabilities[-1] = probabilities[-1] + 1 - probabilities.sum()


    # print(probabilities)

    # Flatten the matrix and probabilities for sampling
    flat_matrix = matrix.flatten()
    flat_probabilities = probabilities.flatten()

    # print(len(flat_matrix))
    # if flat_probabilities.sum()==1:
    #     print('yes')
    # else:
    #     print('no')
    #     print(1-flat_probabilities.sum(),flat_probabilities.sum()-1)
    # print(flat_probabilities.sum())
    # print(flat_probabilities)
    # Sample elements without replacement
    sampled_indices = np.random.choice(
        np.arange(len(flat_matrix)),
        size=sample_size,
        replace=replacement,
        # p=flat_probabilities
    )

    # Create the output matrix with 0s
    output_matrix = np.zeros_like(matrix)  #zeros_like,ones_like
    # output_matrix_mean = scaler.mean_
    # output_matrix_scala = scaler.scale_
    # #     output_matrix_mean = np.array([output_matrix_mean])
    # for i in range(0, len(output_matrix)):
    #     output_matrix[i] = output_matrix_mean[i] - 2 * output_matrix_scala[i]
    # Set the sampled elements to their original values
    np.put(output_matrix, sampled_indices, flat_matrix[sampled_indices])
    mask_num=9
    if mask_num not in sampled_indices:
        output_matrix[mask_num]= -2 #-1.453165
        #print("sex not in")
    return torch.Tensor(output_matrix).cuda()


def sample_operation(dp_sample):

    #
    # if dp_sample == 1:  # 1 without replacement
    #     replacement = False
    #     sampled_matrix = dp_sampling(new_data[i], args.epsilon, args.dp_sampling_size, replacement)
    #     new_data_re.append(sampled_matrix)
    # elif dp_sample == 2:  # 2 with replacement
    #     replacement = True
    #     sampled_matrix = dp_sampling(new_data[i], args.epsilon, args.dp_sampling_size, replacement)
    #     new_data_re.append(sampled_matrix)
    # else:
    #     new_data_re.append(new_data[i])

    return 0


def vib_grad_infer_train(dataset, model, loss_fn, infer_classifier, infer_classifier_of_grad, args, epoch, mu_list, sigma_list, train_loader,train_bs_begin):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_inf = torch.optim.Adam(infer_classifier.parameters(), lr=args.lr)
    optimizer_inf_grad = torch.optim.Adam(infer_classifier_of_grad.parameters(), lr=args.lr)
    for step, (x, y, inf_label) in enumerate(dataset):
        x, y, inf_label= x.to(args.device), y.to(args.device), inf_label.to(args.device) # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)

        logits_z, logits_y, mu, logvar = model(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q = loss_fn(logits_y, y)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        #x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        #BCE = reconstruction_function(x_hat, x)  # mse loss for vae

        loss = args.beta * KLD_mean + H_p_q #args.beta * KLD_mean +
        #loss = args.beta * KLD_mean + BCE  # / (args.batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()


        logits_y_inf = infer_classifier(logits_z.detach())  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q_inf = loss_fn(logits_y_inf, inf_label)

        loss_inf = H_p_q_inf
        optimizer_inf.zero_grad()
        loss_inf.backward()
        torch.nn.utils.clip_grad_norm_(infer_classifier.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer_inf.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == 'encoder.fc2.weight':
                    fc2_grad = param.grad
                    fc2_grad = fc2_grad.reshape((1,-1))
                    # print(fc2_grad.shape)
                # if param.grad ==None: continue
                # print(name, param.grad.shape)

        # break

        logits_y_inf_grad = infer_classifier_of_grad(fc2_grad.detach())  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        #used to train infer model from grad
        H_p_q_inf_grad = loss_fn(logits_y_inf_grad, inf_label)

        loss_inf_grad = H_p_q_inf_grad
        optimizer_inf_grad.zero_grad()
        loss_inf_grad.backward()
        torch.nn.utils.clip_grad_norm_(infer_classifier.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer_inf_grad.step()

        acc_inf_grad = (logits_y_inf_grad.argmax(dim=1) == inf_label).float().mean().item()
        acc_inf = (logits_y_inf.argmax(dim=1) == inf_label).float().mean().item()
        acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'acc_inf': acc_inf,
            'loss': loss.item(),
            #'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }
        train_bs_begin=train_bs_begin+1
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            # print(x)
            # print(y)
            # x_hat_cpu = x_hat.cpu().data
            # x_hat_cpu = x_hat_cpu.clamp(0, 1)
            # x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
            # grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

    return model, mu_list, sigma_list, infer_classifier, infer_classifier_of_grad, train_bs_begin



def print_multal_info(vibi_f_frkl_ss, dataloader_erase,dataloader_sampled, args):
    reconstructor = LinearModel(n_feature=args.dimZ, n_output=14)
    reconstructor = reconstructor.to(args.device)
    optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)


    reconstruction_function = nn.MSELoss(size_average=False)

    #
    # final_round_mse = []
    # for epoch in range(args.num_epochs):
    #     vibi_f_frkl_ss.train()
    #     step_start = epoch * len(dataloader_erase)
    #     for step, (x, y) in enumerate(dataloader_erase, start=step_start):
    #         x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    #         x = x.view(x.size(0), -1)
    #         logits_z, logits_y, x_hat, mu, logvar = vibi_f_frkl_ss(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
    #
    #         x_hat = torch.sigmoid(reconstructor(logits_z))
    #         x_hat = x_hat.view(x_hat.size(0), -1)
    #         x = x.view(x.size(0), -1)
    #         # x = torch.sigmoid(torch.relu(x))
    #         BCE = reconstruction_function(x_hat, x)  # mse loss
    #         loss = BCE
    #
    #         optimizer_recon.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(vibi_f_frkl_ss.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
    #         optimizer_recon.step()
    #
    #         if epoch == args.num_epochs - 1:
    #             final_round_mse.append(BCE.item())
    #         if step % len(dataloader_erase) % 600 == 0:
    #             print("loss", loss.item(), 'BCE', BCE.item())
    #
    # print("final_round mse rfu_ss", np.mean(final_round_mse))

    M = Mine1(noise_size=args.dimZ, sample_size=14)
    M.to(args.device)
    M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)
    step_start=0
    mutual_training_round = int (0.1 / args.erased_local_r) + 1
    t_round =  int( len(dataloader_erase) / args.erased_local_r * 0.1)
    print(t_round, len(dataloader_erase))
    for i in range(mutual_training_round):
        #for step, (x, y) in enumerate(dataloader_erase, start=step_start):
        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_sampled):
            if x.size(0)!=args.batch_size: continue
            t_round = t_round - 1
            if t_round < 0 : break
            x, y = x.to(args.device), y.to(args.device)
            x = x.view(x.size(0), -1)
            x2, y2 = x2.to(args.device), y2.to(args.device)
            x2 = x2.view(x2.size(0), -1)
            logits_z, logits_y, mu, logvar = vibi_f_frkl_ss(x2, mode='distribution')  # (B, C* h* w), (B, N, 10)

            B, Z_size = logits_z.shape
            x = x.view(x.size(0), -1)
            mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
            while mi < 0:
                mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
            if t_round % 100 == 0:
                print(t_round, ' mutual info ', mi)


    print('mutual information after vibi_f_frkl_ss unlearning', mi)
    print()
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
    print('kld_mean', KLD_mean.item())
    return mi

    # for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
    #     x, y = x.to(args.device), y.to(args.device)
    #     x = x.view(x.size(0), -1)
    #
    #     x2, y2 = x2.to(args.device), y2.to(args.device)
    #     x2 = x2.view(x2.size(0), -1)
    #
    #     logits_z, logits_y, x_hat, mu, logvar = vibi_f_frkl_ss(x2, mode='forgetting')  # (B, C* h* w), (B, N, 10)
    #
    #     B, Z_size = logits_z.shape
    #     for i in range(10):
    #         mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
    #         if i % 1 == 0:
    #             print('mutual info', mi)
    #     print()
    #     print('mutual information after vibi_f_frkl_ss unlearning', mi)
    #     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    #     KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    #     KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
    #     print('kld_mean', KLD_mean.item())
    #     break




seed = 1 # 0, 1,2,3,7

torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# parse args
args = args_parser()
args.gpu = 0
# args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.model = 'z_linear'
args.num_epochs = 5
args.dataset = 'Adult'
args.add_noise = False
args.beta = 0.001  # 0.001
args.lr = 0.001
args.max_norm=1
args.dimZ = 9 #9 #40 #2 9 , 12
args.batch_size =20  # 1 for infer ,20 for normal training
args.erased_local_r = 0.06  # the erased data ratio
args.back_acc_threshold = 0.1

args.mi_rate = 1 #args.beta
args.epsilon = 1.0
args.dp_sampling_size = int(14*0.601)

args.mi_epoch = 40
args.SNR = 6
args.kld_to_org = 1
args.unlearn_bce = 0.1  # beta_u  0.1
# args.self_sharing_rate = 1

args.unl_r_for_bayesian = args.unlearn_bce
args.hessian_rate = 0.005

args.unlearn_learning_rate = 1
args.reverse_rate = 0.5
args.kld_r = 1
args.unlearn_ykl_r = args.unlearn_learning_rate*0.4
args.unlearn_bce_r = args.unlearn_learning_rate
args.unl_r_for_bayesian = args.unlearn_learning_rate
args.self_sharing_rate = args.unlearn_learning_rate*5   # compensation, small will perform better in erasure
args.unl_conver_r = 2
#args.hessian_rate = 0.005
args.hessian_rate = 0.00005

# print('args.beta', args.beta, 'args.lr', args.lr)

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = args.device
print("device", device)


if args.dataset == 'MNIST':
    transform = T.Compose([
        T.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(), ])
    train_set = MNIST('../../data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('../../data/mnist', train=False, transform=trans_mnist, download=True)
    train_set_no_aug = train_set
elif args.dataset == 'CIFAR10':
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.ToTensor(),
                                 ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
    test_transform = T.Compose([T.ToTensor(),
                                ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    train_set = CIFAR10('../../data/cifar', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('../../data/cifar', train=False, transform=test_transform, download=True)
    train_set_no_aug = CIFAR10('../../data/cifar', train=True, transform=test_transform, download=True)
elif args.dataset == 'Adult':
    # Load the dataset.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                    "income"]
    data = pd.read_csv(url, names=column_names, na_values='?')

    # Drop missing values.
    data = data.dropna()

    # Select features and target.
    X = data.drop(columns=['income'])
    y = data['income']


    # Convert categorical variables to numeric.
    for col in X.columns:
        if X[col].dtype == object:
            le = preprocessing.LabelEncoder()
            X[col] = le.fit_transform(X[col])


    # Convert target to binary.
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    scaler = StandardScaler() # StandardScaler , MinMaxScaler

    # Scale the features

    # Split the data into training set and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    infer_sex = X_train['sex']
    print(infer_sex)
    #print(X_train)
    # Get feature names from training data
    feature_names_train = X_train.columns.tolist()

    # Get feature names from test data
    feature_names_test = X_test.columns.tolist()

    print(feature_names_train)
    print(feature_names_test)
    # # Reset index
    # X_train = X_train.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)

    # Identify the samples where y=1
    #selected_indices = y_train[y_train == 1].index
    selected_indices = np.where(y_train == 1)[0]

    # X_train.iloc[selected_indices].copy()
    # Shuffle the indices and select the first 1000
    #np.random.shuffle(selected_indices)
    #print(selected_indices)
    #print()
    #selected_indices = selected_indices[:1000]
    print('lenght of Xtrain',len(X_train))
    erased_size = int(len(X_train)*args.erased_local_r)
    print('length of erased', erased_size)
    selected_indices = np.random.choice(selected_indices, size=erased_size, replace=False)
    # print(selected_indices)
    # Create the 'backdoored' data
    y_backdoor = y_train[selected_indices].copy()
    y_backdoor = y_backdoor - 1
    print('y_backdoor',y_backdoor)
    X_backdoor =  X_train.iloc[selected_indices].copy()

    # Change the 'education_num' feature to 0 in X_backdoor
    X_backdoor['education-num'] = 2  # max = 12?

    #
    # print(X_backdoor)
    y_backdoor_inf = X_backdoor['sex']
    print(y_backdoor_inf)
    print(len(y_backdoor_inf), np.average(y_backdoor_inf))
    print("len y",len(y), np.average(y))
    print("ken x back", len(X_backdoor))
    # Change the y value to 0 in y_backdoor
    y_backdoor[:] = 0

    X_remaining = X_train
    y_remaining = y_train

    # Append the backdoored data to the training data.
    X_train = pd.concat([X_train, X_backdoor])
    y_train = np.hstack((y_train, y_backdoor))

    # Get feature names from training data
    feature_names_train = X_train.columns.tolist()

    # Get feature names from test data
    feature_names_test = X_test.columns.tolist()

    print(feature_names_train)
    print(feature_names_test)
    # Check if any feature is missing in the test data
    missing_features = [feature for feature in feature_names_train if feature not in feature_names_test]

    print("Missing features:", missing_features)

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
    X_remaining = pd.DataFrame(scaler.transform(X_remaining), columns=X_train.columns)
    X_backdoor = pd.DataFrame(scaler.transform(X_backdoor), columns=X_train.columns)

    # Convert the data to PyTorch tensors.
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    X_backdoor_tensor = torch.tensor(X_backdoor.values, dtype=torch.float)
    # print('backdoor tensor')
    # print(X_backdoor_tensor)
    print('backdoor ')
    # print(X_backdoor_tensor.shape)
    dp_sample_w_o = torch.empty(0, 14).float().to(args.device)
    for i in range(len(X_backdoor_tensor)):
        # print(X_backdoor_tensor[i])
        replacement = False
        temp_dt = dp_sampling(X_backdoor_tensor[i], args.epsilon, args.dp_sampling_size, replacement)
        # print(X_backdoor_tensor)
        # print(temp_dt.shape)
        temp_dt = temp_dt.reshape((1,14))
        dp_sample_w_o = torch.cat([dp_sample_w_o, temp_dt], dim=0)


    dp_sample_w = torch.empty(0, 14).float().to(args.device)
    for i in range(len(X_backdoor_tensor)):
        # print(X_backdoor_tensor[i])
        replacement = True
        temp_dt = dp_sampling(X_backdoor_tensor[i], args.epsilon, args.dp_sampling_size, replacement) # our methods do not have the sampling strategies
        # print(X_backdoor_tensor)
        temp_dt = temp_dt.reshape((1, 14))
        dp_sample_w = torch.cat([dp_sample_w, temp_dt], dim=0)

    # print(dp_sample_w.shape,X_backdoor_tensor.shape)

    y_backdoor_tensor = torch.tensor(y_backdoor, dtype=torch.long)
    X_remaining_tensor = torch.tensor(X_remaining.values, dtype=torch.float)
    y_remaining_tensor = torch.tensor(y_remaining, dtype=torch.long)
    infer_sex_tensor = torch.tensor(infer_sex, dtype=torch.long)
    y_backdoor_inf_tensor = torch.tensor(y_backdoor_inf.values, dtype=torch.long)

    print('lenght of Xtrain', len(X_train))
    print('lenght of Xbackdoor', len(X_backdoor))
    print('lenght of Xremaining', len(X_remaining))
    print(y_backdoor)
    print(y_remaining)
    train_set = Data.TensorDataset(X_train_tensor, y_train_tensor)
    test_set = Data.TensorDataset(X_test_tensor, y_test_tensor)
    backdoor_set = Data.TensorDataset(X_backdoor_tensor, y_backdoor_tensor)
    backdoor_set_sp_w = Data.TensorDataset(dp_sample_w, y_backdoor_tensor)
    backdoor_set_sp_wo = Data.TensorDataset(dp_sample_w_o, y_backdoor_tensor)
    backdoor_set_sp_w_inf = Data.TensorDataset(dp_sample_w, y_backdoor_tensor, y_backdoor_inf_tensor)
    backdoor_set_sp_wo_inf = Data.TensorDataset(dp_sample_w_o, y_backdoor_tensor,y_backdoor_inf_tensor)
    backdoor_set_inf = Data.TensorDataset(X_backdoor_tensor, y_backdoor_tensor, y_backdoor_inf_tensor)
    remaining_set = Data.TensorDataset(X_remaining_tensor, y_remaining_tensor)
    set_with_infer = Data.TensorDataset(X_remaining_tensor, y_remaining_tensor, infer_sex_tensor)



train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
erased_loader = DataLoader(backdoor_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
remaining_loader = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
infer_loader = DataLoader(set_with_infer, batch_size=args.batch_size, shuffle=True, num_workers=0)
backdoor_inf_loader = DataLoader(backdoor_set_inf, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_w_inf = DataLoader(backdoor_set_sp_w_inf, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_wo_inf = DataLoader(backdoor_set_sp_wo_inf, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_w = DataLoader(backdoor_set_sp_w, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_wo = DataLoader(backdoor_set_sp_wo, batch_size=args.batch_size, shuffle=True, num_workers=0)

dataloader_full = DataLoader(train_set, batch_size=20, shuffle=True)

test_dateloader = DataLoader(test_set, batch_size=20, shuffle=True)


vib, lr = init_vib(args)
vib.to(args.device)

loss_fn = nn.CrossEntropyLoss()

reconstruction_function = nn.MSELoss(size_average=True)


acc_test = []
print("learning")

print('Training VIBI')
print(f'{type(vib.encoder).__name__:>10} encoder params:\t{num_params(vib.encoder) / 1000:.2f} K')
print(f'{type(vib.approximator).__name__:>10} approximator params:\t{num_params(vib.approximator) / 1000:.2f} K')
print(f'{type(vib.decoder).__name__:>10} decoder params:\t{num_params(vib.decoder) / 1000:.2f} K')
# inspect_explanations()



infer_classifier = LinearModel(n_feature=args.dimZ, n_output=2).to(args.device)
infer_classifier_of_grad = LinearModel(n_feature=96*args.dimZ*2, n_output=2).to(args.device)


eva_of_vib_back = eva_vib(vib, erased_loader, args, name='vib backdoor adult evaluation')
# train VAE
mu_list = []
sigma_list = []
back_g_acc_lsit = []

train_bs_begin=0
for epoch in range(args.num_epochs):
    vib.train()
    vib, mu_list, sigma_list, infer_classifier, infer_classifier_of_grad, train_bs_begin = vib_infer_train(infer_loader, vib, loss_fn, infer_classifier, infer_classifier_of_grad, args, epoch, mu_list, sigma_list, train_loader,train_bs_begin)
    # acc = eva_vib_inf_grad(vib, infer_classifier_of_grad, backdoor_inf_loader, args, name='grad_infer', epoch=999)
    #acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor', epoch=epoch)
    #back_g_acc_lsit.append(acc_back_g)

print('infer train bs', train_bs_begin)
print('train infer w, at original infer dataset based model')
acc = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_w_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader

print("train infer wo, at original infer dataset based model")
acc = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_wo_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader


# here to trian the infer grad, we need to set the batch size = 1
# acc = eva_vib_inf_grad(copy.deepcopy(vib).to(args.device), infer_classifier_of_grad, infer_loader, args, name='grad_infer_normal', epoch=999)
#
# acc = eva_vib_inf_grad(copy.deepcopy(vib).to(args.device), infer_classifier_of_grad, backdoor_inf_loader, args, name='grad_infer', epoch=999)



train_type = 'VIB'
# train_bs_begin = 0
total_training_time = 0
for epoch in range(args.num_epochs):
    vib.train()
    start_time = time.time()
    vib, mu_list, sigma_list, train_bs_begin = vib_train(train_loader, vib, loss_fn, reconstruction_function, args, epoch, mu_list, sigma_list, train_loader, train_bs_begin, train_type)
    #acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor', epoch=epoch)
    #back_g_acc_lsit.append(acc_back_g)
    #
    # start_time = time.time()
    end_time = time.time()

    running_time = end_time - start_time
    total_training_time += running_time
    print(f'VIB one big round Training took {running_time} seconds')

print()
print("total training time ", total_training_time*2)# we use total_training_time*2, because the former the vib model is also trained during infer model training
total_training_time_of_org = total_training_time*2
print('infer train bs', train_bs_begin)
train_bs_of_org = train_bs_begin
#acc = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_wo_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader



print('train infer w')
infer_acc_w = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_w_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader

print("train infer wo")
infer_acc_wo = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_wo_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader



print('train vib used ',train_bs_begin)
print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))




eva_of_vib_of_org = eva_vib(vib, test_dateloader, args, name='vib adult evaluation')

eva_of_vib_back_of_org = eva_vib(vib, erased_loader, args, name='vib backdoor adult evaluation')



reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
reconstructor = reconstructor.to(device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)

reconstruction_function = nn.MSELoss(size_average=False)

print()
print("start npis unlearn")
#dataloader_dp_sampled

start_time = time.time()

vibi_f_frkl_nips, optimizer_frkl_nips, valid_acc_nipsu,  backdoor_acc_nipsu, train_bs_nipsu = unlearning_frkl_train(copy.deepcopy(vib).to(args.device), erased_loader,
                                                          remaining_loader, loss_fn,
                                                          reconstructor,
                                                          reconstruction_function, test_loader, train_loader, train_type='NIPSU')

eva_of_vib_back = eva_vib(vibi_f_frkl_nips, erased_loader, args, name='unlearned nips backdoor adult evaluation')

#
#start_time = time.time()
end_time = time.time()

running_time = end_time - start_time
print(f'NIPS Training took {running_time} seconds')

print()
print("start VIBU-SS with dp sampling drop with replacement")
#dataloader_dp_sampled

start_time = time.time()
vibi_f_frkl_sampled_w, optimizer_frkl_sampled_w, valid_acc_mcfu_w,  backdoor_acc_mcfu_w, train_bs_mcfu_w  = unlearning_frkl_train(copy.deepcopy(vib).to(args.device), erased_loader_sp_w,
                                                          remaining_loader, loss_fn,
                                                          reconstructor,
                                                          reconstruction_function, test_loader, train_loader, train_type='VIBU-SS') #VIBU-SS

backdoor_acc_mcfu_w = eva_vib(vibi_f_frkl_sampled_w, erased_loader, args, name='unlearned vib w_sampled backdoor adult evaluation')
valid_acc_mcfu_w = eva_vib(vibi_f_frkl_sampled_w, test_loader, args, name='unlearned vib w_sampled normal adult evaluation')

#
#start_time = time.time()
end_time = time.time()

running_time = end_time - start_time
print(f'sampled_w Training took {running_time} seconds')


print()
print("start VIBU-SS with dp sampling drop without replacement")
#dataloader_dp_sampled

vibi_f_frkl_sampled_wo, optimizer_frkl_sampled_wo, valid_acc_mcfu_wo,  backdoor_acc_mcfu_wo, train_bs_mcfu_wo = unlearning_frkl_train(copy.deepcopy(vib).to(args.device), erased_loader_sp_wo,
                                                          remaining_loader, loss_fn,
                                                          reconstructor,
                                                          reconstruction_function, test_loader, train_loader, train_type='VIBU-SS')

backdoor_acc_mcfu_wo = eva_vib(vibi_f_frkl_sampled_wo, erased_loader, args, name='unlearned vib w/o_sampled backdoor adult evaluation')
valid_acc_mcfu_wo = eva_vib(vibi_f_frkl_sampled_wo, test_loader, args, name='unlearned vib w/o_sampled normal adult evaluation')




print('mutual w info for vibi')
mi_of_w = print_multal_info(copy.deepcopy(vib).to(args.device), erased_loader, erased_loader_sp_w, args)

print('mutual wo info for vibi')
mi_of_wo = print_multal_info(copy.deepcopy(vib).to(args.device), erased_loader, erased_loader_sp_wo, args)

learn_model_type = 'nips'
vib_nips, lr = init_vib(args)
vib_nips.to(args.device)
train_type = 'NIPS'
train_bs_begin = 0
for epoch in range(args.num_epochs):
    vib.train()
    vib_nips, mu_list, sigma_list, train_bs_begin= vib_train(train_loader, vib_nips, loss_fn, reconstruction_function, args, epoch, mu_list, sigma_list, train_loader, train_bs_begin, train_type)
    #acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor', epoch=epoch)
    #back_g_acc_lsit.append(acc_back_g)

print('nips trian bs', train_bs_begin)
print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))


eva_of_vib = eva_vib(vib_nips, test_dateloader, args, name='vib_nips adult evaluation')
print('mutual info for vibi_for_nips')
mi_of_vbu = print_multal_info(vib_nips, erased_loader, erased_loader, args)

#print('mutual vibi_f_hessian for vibi_for_nips') # to test hessian-based unlearning, run the backdoor_FedHessian2_temp.py
#print_multal_info(vibi_f_hessian, dataloader_erase,dataloader_erase, args)

# print_multal_info(copy.deepcopy(vibi_f_hessian).to(args.device), dataloader_erase,dataloader_erase, args)
# print_multal_info_for_hessian(copy.deepcopy(vibi_f_hessian).to(args.device), dataloader_erase,dataloader_erase, args)

# print('gradient vibi')
# print_multal_info_for_hessian(copy.deepcopy(vibi).to(args.device), dataloader_erase, dataloader_erase, args)


# print('mutual vibi_f_nipsu for vibi_for_nips')
# print_multal_info(vibi_f_nipsu, dataloader_erase, dataloader_erase, args)

print('mutual vibi_f_frkl_sampled_w for vibi_f_frkl_sampled_w')
mi_of_mcfu_w_after_unl = print_multal_info(vibi_f_frkl_sampled_w, erased_loader, erased_loader, args)

print('mutual vibi_f_frkl_sampled for vibi_f_frkl_sampled w/o')
mi_of_mcfu_w_after_unl = print_multal_info(vibi_f_frkl_sampled_wo, erased_loader, erased_loader, args)






args.batch_size=1

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
erased_loader = DataLoader(backdoor_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
remaining_loader = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
infer_loader = DataLoader(set_with_infer, batch_size=args.batch_size, shuffle=True, num_workers=0)
backdoor_inf_loader = DataLoader(backdoor_set_inf, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_w_inf = DataLoader(backdoor_set_sp_w_inf, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_wo_inf = DataLoader(backdoor_set_sp_wo_inf, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_w = DataLoader(backdoor_set_sp_w, batch_size=args.batch_size, shuffle=True, num_workers=0)
erased_loader_sp_wo = DataLoader(backdoor_set_sp_wo, batch_size=args.batch_size, shuffle=True, num_workers=0)

dataloader_full = DataLoader(train_set, batch_size=20, shuffle=True)

test_dateloader = DataLoader(test_set, batch_size=20, shuffle=True)


vib, lr = init_vib(args)
vib.to(args.device)

loss_fn = nn.CrossEntropyLoss()

reconstruction_function = nn.MSELoss(size_average=True)


acc_test = []

infer_classifier = LinearModel(n_feature=args.dimZ, n_output=2).to(args.device)
infer_classifier_of_grad = LinearModel(n_feature=96*args.dimZ*2, n_output=2).to(args.device)


eva_of_vib_back = eva_vib(vib, erased_loader, args, name='vib backdoor adult evaluation')
# train VAE
mu_list = []
sigma_list = []
back_g_acc_lsit = []

train_bs_begin=0
for epoch in range(args.num_epochs):
    vib.train()
    vib, mu_list, sigma_list, infer_classifier, infer_classifier_of_grad, train_bs_begin = vib_grad_infer_train(infer_loader, vib, loss_fn, infer_classifier, infer_classifier_of_grad, args, epoch, mu_list, sigma_list, train_loader,train_bs_begin)
    # acc = eva_vib_inf_grad(vib, infer_classifier_of_grad, backdoor_inf_loader, args, name='grad_infer', epoch=999)
    #acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor', epoch=epoch)
    #back_g_acc_lsit.append(acc_back_g)

print('infer train bs', train_bs_begin)
# print('train infer w, at original infer dataset based model')
# acc = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_w_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader
#
# print("train infer wo, at original infer dataset based model")
# acc = eva_vib_inf(copy.deepcopy(vib).to(args.device), infer_classifier, erased_loader_sp_wo_inf, args, name='infer', epoch=999) # erased_loader_sp_wo_inf, backdoor_inf_loader


# here to trian the infer grad, we need to set the batch size = 1
infer_acc_grad = eva_vib_inf_grad(copy.deepcopy(vib).to(args.device), infer_classifier_of_grad, infer_loader, args, name='grad_infer_normal', epoch=999)

infer_backdoor_acc_grad = eva_vib_inf_grad(copy.deepcopy(vib).to(args.device), infer_classifier_of_grad, backdoor_inf_loader, args, name='grad_infer', epoch=999)



### we print the main result of this experiment on the overall evaluation table here. For some other values about HBFU, we need to run the FedHessian file.

print('----------------------------------------------------')
print('we print the main result of this experiment on the overall evaluation table here. For some other values about HBFU, we need to run the FedHessian file.')
# print the results about Origin
print()
print("print the results about Origin")
print("Mutual information of Origin:",) #mi_of_grad_hbfu
print("Privacy leak attacks of Origin (Acc):", infer_acc_grad) #final_round_mse_on_whole_test_set
print("Backdoor Acc. of Origin:", eva_of_vib_back_of_org)
print("Acc. on test dataset of Origin:", eva_of_vib_of_org)
print("Running time (s) of Origin:", total_training_time_of_org)



avg_of_one_batch = total_training_time_of_org/train_bs_of_org


# print the results about HBFU
print()
print("print the results about HBFU")
print("Mutual information of HBFU:", "run FedHessian to get")
print("Privacy leak attacks of HBFU (Acc):", infer_backdoor_acc_grad) #final_round_mse_on_only_erased_set
print("Backdoor Acc. of HBFU:", "run FedHessian to get")
print("Acc. on test dataset of HBFU:", "run FedHessian to get")
print("Running time (s) of HBFU:", "run FedHessian to get")


# print the results about VBU
print()
print("print the results about VBU")
print("Mutual information of VBU:", mi_of_vbu)
print("Privacy leak attacks of VBU (Acc):", 100)  # because the server can get the erased samples on to implement VBU, therefore the recover mse is 0
print("Backdoor Acc. of VBU:", backdoor_acc_nipsu)
print("Acc. on test dataset of VBU:", valid_acc_nipsu)
print("Running time (s) of VBU:", train_bs_nipsu*avg_of_one_batch)



# print the results about PriMU_w
print()
print("print the results about PriMU_w")
print("Mutual information of PriMU_w:", mi_of_w )
print("Privacy leak attacks of PriMU_w:", infer_acc_w)
print("Backdoor Acc. of PriMU_w:", backdoor_acc_mcfu_w)
print("Acc. on test dataset of PriMU_w:", valid_acc_mcfu_w)
print("Running time (s) of PriMU_w:", train_bs_mcfu_w * avg_of_one_batch*2) # as the contain the auxiliary dataset



# print the results about PriMU_w/o
print()
print("print the results about PriMU_w/o")
print("Mutual information of PriMU_w/o:", mi_of_wo)
print("Privacy leak attacks of PriMU_w/o:", infer_acc_wo)
print("Backdoor Acc. of PriMU_w/o:", backdoor_acc_mcfu_wo)
print("Acc. on test dataset of PriMU_w/o:", valid_acc_mcfu_wo)
print("Running time (s) of PriMU_w/o:", train_bs_mcfu_wo * avg_of_one_batch*2)




'''
On Adult, EDR = 6%, \\beta = 0.001, SR = 60%

| On Adult             | Origin       | HBFU     |    VBU   |  PriMU_w  | PriMU_w/o |
| --------             | --------     | -------- | -------- | -------- | -------- |
| Mutual information   | 2.14         | 3.66     | 10.99    | 3.48     | 2.74     |
| Privacy leak attacks | 90.76% (Acc.)| 96.80%   | 99.99%   | 57.68%   | 59.16%   |
| Backdoor Acc.        | 99.99%       | 99.99%   | 8.45%    | 9.41%    | 7.04%    |
| Acc. on test dataset | 85.57%       | 85.45%   | 64.64%   | 84.34%   | 84.05%   |
| Running time (s)     | 43.67        | 85.90    | 0.13     | 1.78     | 1.30     |

'''

# # Evaluate the model.
# vib.eval()  # Set the model to evaluation mode.
# with torch.no_grad():  # Don't track gradients.
#
#     for step, (x, y) in enumerate(test_dateloader):
#         x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
#         x = x.view(x.size(0), -1)
#
#         logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='distribution')  # (B, C* h* w), (B, N, 10)
#         # VAE two loss: KLD + MSE
#     outputs = classifier(X_test_tensor.cuda())#.squeeze()
#     #predicted = (torch.sigmoid(outputs) > 0.5).long()
#     _, predicted = torch.max(outputs, 1)
#     #y_test_tensor = y_test_tensor.reshape((y_test_tensor.size(0), 1))  # images.reshape((B, -1))
#     correct = (predicted == y_test_tensor.cuda()).sum().item()
#     total = y_test_tensor.shape[0]
#     accuracy = correct / total
#     print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
# dp_sample_w = torch.empty(0, 14).float().to(args.device)
# for i in range(len(X_backdoor_tensor)):
#     # print(X_backdoor_tensor[i])
#     replacement = True
#     temp_dt = dp_sampling(X_backdoor_tensor[i], args.epsilon, args.dp_sampling_size, replacement)
#     # print(X_backdoor_tensor)
#     temp_dt = temp_dt.reshape((1, 14))
#     dp_sample_w = torch.cat([dp_sample_w, temp_dt], dim=0)
#     print(temp_dt)
#     break
