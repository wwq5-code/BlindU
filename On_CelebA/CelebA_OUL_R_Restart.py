import sys

sys.argv = ['']
del sys

import math

import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
import torchvision
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100, CelebA
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import copy
import random
import time
from torch.nn.functional import cosine_similarity




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


def add_trigger_new(add_backdoor, dataset, poison_samples_size, mode):
    print("## generate——test" + mode + "Bad Imgs")

    # indices = dataset.indices
    list_from_dataset_tuple = list(dataset)
    list_from_dataset_tuple_target = list(dataset)
    indices = list(range(len(list_from_dataset_tuple)))
    new_data_re = []

    x, y = list_from_dataset_tuple[0]

    # total_poison_num = int(len(new_data) * portion/10)
    _, width, height = x.shape

    for i in range(len(list_from_dataset_tuple)):
        if add_backdoor == 1:

            x, y = list_from_dataset_tuple[i]
            # image_steg = generate_gray_laplace_small_trigger_noise(new_data[i])
            # new_data[i] = image_steg

            # Plotting
            # plt.imshow(embedded_image)
            # plt.title("Image with Embedded Feature Map")
            # plt.axis('off')
            # plt.show()

            # add trigger as general backdoor
            # x[:, width - 3, height - 3] = 1
            # x[:, width - 3, height - 4] = 1
            # x[:, width - 4, height - 3] = 1
            # x[:, width - 4, height - 4] = 1
            temp = 1 - x[:, -5:-2, -5:-2]
            x[:, -5:-2, -5:-2] = x[:, -5:-2, -5:-2] + temp * args.laplace_scale

            list_from_dataset_tuple_target[i] = x, y

            y[20]=1-y[20]
            list_from_dataset_tuple[i] = x, y

            # new_data[i, :, width - 23, height - 21] = 254
            # new_data[i, :, width - 23, height - 22] = 254
            # new_data[i, :, width - 22, height - 21] = 254
            # new_data[i, :, width - 24, height - 21] = 254
            # new_data[i] = torch.from_numpy(embedded_image).view([1,28,28])
            #      new_data_re.append(embedded_image)
            poison_samples_size = poison_samples_size - 1
            if poison_samples_size <= 0:
                break
        # x=torch.tensor(new_data[i])
        # x_cpu = x.cpu().data
        # x_cpu = x_cpu.clamp(0, 1)
        # x_cpu = x_cpu.view(1, 1, 28, 28)
        # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
        # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        # plt.show()

    # print(len(new_data_re))
    return Subset(list_from_dataset_tuple, indices), Subset(list_from_dataset_tuple_target, indices)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Start with a linear layer to get the correct number of features
        self.fc = nn.Linear(128, 512)

        # Upscale to the desired dimensions using transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # Output: 256 x 2 x 2
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # Output: 128 x 4 x 4
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Output: 64 x 8 x 8
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # Output: 32 x 16 x 16

        # Final layer to produce an output of 3 channels (CIFAR-10 image)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # Output: 3 x 32 x 32

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid for final layer to output values in [0, 1]

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(-1, 512, 1, 1)  # Reshape to start the transposed convolutions
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.sigmoid(self.deconv5(x))  # Use sigmoid if the image values are normalized between 0 and 1
        return x

class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
        self.fc3 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)  # output

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
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == '64QAM_distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print(logits_z)

            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar

        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 2))  # (B,   10)
            x_hat = self.reconstruction(logits_z, x)
            return logits_z, logits_y, x_hat, mu, logvar

        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # VAE is not related to labels
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # logits_y = self.approximator(logits_z)  # (B , 10)
            # logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z, x)
            return logits_z, x_hat, mu, logvar
        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y

    def reconstruction(self, logits_z, x):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.decoder(logits_z)
        x_v = x.view(x.size(0), -1)
        output_x = output_x.view(output_x.size(0), -1)
        x2 = F.relu(x_v - output_x)
        return torch.sigmoid(self.fc3(x2))

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
        encoder = resnet18(1, args.dimZ * 2)  # 64QAM needs 6 bits
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

    elif args.dataset == 'CelebA':
        approximator = LinearModel(n_feature=args.dimZ, n_output=2)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = Decoder() #LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def vib_train(dataset, model, loss_fn, reconstruction_function, args, epoch, train_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y[:,20].to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        x.requires_grad = True
        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        H_p_q = loss_fn(logits_y, y)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # mse loss for vae # torch.mean((x_hat - x) ** 2 * (x_inverse_m > 0).int()) / 0.75 # reconstruction_function(x_hat, x_inverse_m)  # mse loss for vae
        BCE = reconstruction_function(x_hat, x)
        # Calculate the L2-norm

        loss = args.beta * KLD_mean + H_p_q # + args.mse_rate * BCE

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        # Check if gradients exist for x
        # input_gradient = x.grad.detach()
        optimizer.step()

        # acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            # 'acc': acc,
            'loss': loss.item(),
            # 'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }
        # if epoch == args.num_epochs - 1:
        #     mu_list.append(torch.mean(mu).item())
        #     sigma_list.append(sigma)
        if step % len(dataset) % 10000 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(dataset):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            x_cpu = x.cpu().data
            x_cpu = x_cpu.clamp(0, 1)
            x_cpu = x_cpu.view(x_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_cpu, nrow=4)
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

            x_hat_cpu = x_hat.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4)
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()
            print("print x grad")
            # print(input_gradient)

    return model

# here we prepare the unlearned model, and we can calculate the model difference
def prepare_unl(erasing_dataset, model, loss_fn, args, noise_flag):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_test = []
    backdoor_acc_list = []

    print(len(erasing_dataset.dataset))
    for epoch in range(args.num_epochs):
        model.train()
        for step, (x_e, y_e) in enumerate(erasing_dataset):
            x_e, y_e = x_e.to(args.device), y_e[:,20].to(args.device)  # (B, C, H, W), (B, 10)
            if noise_flag =="noise":
                x_e = add_laplace_noise(x_e, epsilon=args.laplace_epsilon, sensitivity=1, args=args)
            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = model(x_e,
                                                                        mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

            KLD_element = mu_e.pow(2).add_(logvar_e.exp()).mul_(-1).add_(1).add_(logvar_e).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
            H_p_q = loss_fn(logits_y_e, y_e)
            loss = args.beta * KLD_mean - args.unlearn_learning_rate * H_p_q
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
            acc_back = (logits_y_e.argmax(dim=1) == y_e).float().mean().item()
            metrics = {
                'acc_back': acc_back,
                'loss1': loss.item(),
                'KLD_mean': KLD_mean.item(),
                # '1-JS(p,q)': JS_p_q,
                'mu': torch.mean(mu_e).item(),
                # 'KLD': KLD.item(),
                # 'KLD_mean': KLD_mean.item(),
            }
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu).item())
            #     sigma_list.append(sigma)
            if step % len(erasing_dataset) % 10000 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{step % len(erasing_dataset):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
                x_cpu = x_e.cpu().data
                x_cpu = x_cpu.clamp(0, 1)
                x_cpu = x_cpu.view(x_cpu.size(0), 3, 32, 32)
                grid = torchvision.utils.make_grid(x_cpu, nrow=4)
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

                print(acc_back)
                print("print x grad")
                # print(updated_x)
            # if acc_back<0.1:
            #     break
        backdoor_acc = eva_vib(model, erasing_dataset, args, name='on erased data', epoch=999)
        if backdoor_acc < 0.1:
            break
    return model


def prepare_update_direction(unlearned_vib, model):
    update_deltas_direction = []
    #     for name, param in model.named_parameters():
    #         print(name)
    for param1, param2 in zip(unlearned_vib.approximator.parameters(), model.approximator.parameters()):
        # Calculate the difference (delta) needed to update model1 towards model2
        if param1.grad is not None:
            delta = param2.data.view(-1) - param1.data.view(-1)
            grad_direction = torch.sign(delta)
            update_deltas_direction.append(grad_direction)

    return update_deltas_direction


# here the dataset is the watermarking dataset, also we need a dataset of the target dataset as target
def construct_input(dataset, unlearned_vib, model, loss_fn, args, epoch):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
    temp_label = torch.empty(0,40).long().to(args.device)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        b_s, _, width, height = x.shape
        init_random_x = torch.zeros_like(x).to(args.device)
        init_random_x[:, :, -5:-2, -5:-2] = args.laplace_scale
        random_patch = init_random_x[:, :, -5:-2, -5:-2]
        break

    for step, (x, y) in enumerate(dataset):
        update_deltas_direction = prepare_update_direction(unlearned_vib, model)
        x, y_20 = x.to(args.device), y[:,20].to(args.device)  # (B, C, H, W), (B, 10)

        # x = x.view(x.size(0), -1)
        b_s, _, width, height = x.shape
        if b_s != args.batch_size:
            continue


        random_patch.requires_grad = True
        optimizer_x = torch.optim.Adam([random_patch], 0.1)
        x2 = x
        x2[:, :, -5:-2, -5:-2] = x2[:, :, -5:-2, -5:-2] + random_patch
        x2 = x2.clamp(0, 1)
        logits_z, logits_y, x_hat, mu, logvar = model(x2, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
        H_p_q_1 = loss_fn(logits_y, y_20)

        loss1 = args.beta * KLD_mean + H_p_q_1  # + similarity_loss + KLD_mean

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        grads1 = [torch.sign(param.grad.view(-1)) for param in model.approximator.parameters() if
                  param.grad is not None]

        # params_model = [p.data.view(-1) for p in model.parameters()]
        p1 = torch.cat(grads1)
        p2 = torch.cat(update_deltas_direction)
        # print(p1.shape, p2.shape)
        cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
        similarity_loss = 1 - cos_sim  # Loss is lower when similarity is higher


        total_loss = loss1 + similarity_loss.detach()

        # Then use total_loss for your optimization step
        #         optimizer.zero_grad()

        optimizer_x.zero_grad()
        total_loss.backward()
        optimizer_x.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        # Check if gradients exist for x
        input_gradient = random_patch.grad.detach()

        l1_norm = torch.norm(input_gradient, p=1)
        epsilon = args.ep_distance
        # piece_v = (epsilon - l1_norm) / (b_s * 4)
        # input_gradient = input_gradient / l1_norm * 16
        x[:, :, -5:-2, -5:-2] = x[:, :, -5:-2, -5:-2] + random_patch.detach()
        updated_x = x.clamp(0, 1)
        input_gradient = input_gradient
        temp_img = torch.cat([temp_img, updated_x.detach()], dim=0)
        y = y.to(args.device)
        temp_label = torch.cat([temp_label, y.detach()], dim=0)
        # optimizer.step()

        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'loss1': loss1.item(),
            'similarity_loss': similarity_loss.item(),
            'KLD_mean': KLD_mean.item(),
            'l1_norm': l1_norm.item(),
            # 'BCE': BCE.item(),
            'H(p,q)': H_p_q_1.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            # 'KLD': KLD.item(),
            # 'KLD_mean': KLD_mean.item(),
        }
        # if epoch == args.num_epochs - 1:
        #     mu_list.append(torch.mean(mu).item())
        #     sigma_list.append(sigma)
        if step % len(dataset) % 10000 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(dataset):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            x_cpu = x.cpu().data
            x_cpu = x_cpu.clamp(0, 1)
            x_cpu = x_cpu.view(x_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_cpu, nrow=4)
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

            x_hat_cpu = updated_x.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4)
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()
            print(acc)
            print("print x grad")
            # print(input_gradient[:, :,  -7:-2, -7:-2])
            # print(updated_x)
    d = Data.TensorDataset(temp_img, temp_label)
    d_loader = DataLoader(d, batch_size=args.batch_size, shuffle=True)
    return d_loader, model


@torch.no_grad()
def eva_vib(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y[:,20].to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

        x_hat = x_hat.view(x_hat.size(0), -1)

        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}')
    return acc


@torch.no_grad()
def eva_vib_target(vib, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()

    num_total = 0
    num_correct = 0
    for batch_idx, (x, y_t, y_ad) in enumerate(dataloader_erase):
        x, y_t, y_ad = x.to(args.device), y_t.to(args.device), y_ad.to(args.device)  # (B, C, H, W), (B, 10)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)

        x_hat = x_hat.view(x_hat.size(0), -1)
        if name == 'on adversarial target':
            if y_ad.ndim == 2:
                y_ad = y_ad.argmax(dim=1)
            num_correct += (logits_y.argmax(dim=1) == y_ad).sum().item()
            num_total += len(x)
        elif name == 'on target':
            if y_t.ndim == 2:
                y_t = y_t.argmax(dim=1)
            num_correct += (logits_y.argmax(dim=1) == y_t).sum().item()
            num_total += len(x)

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, total_num:{num_total}')
    return acc


def train_reconstructor(vib, train_loader, reconstruction_function, args):
    # init reconsturctor
    # reconstructor = LinearModel(n_feature=40, n_output=28 * 28).to(args.device)
    vib.decoder.trainable = False
    vib.fc3.trainable = False

    reconstructor = resnet18(1, args.dimZ).to(args.device)
    optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)
    reconstructor.train()
    epochs = 1

    optimizer_vib_reconstruction = torch.optim.Adam(vib.decoder.parameters(), lr=args.lr)


    ## training epochs
    total_training_samples = 50000*0.06
    erased_samples = 50000*args.erased_local_r
    epochs= int(total_training_samples/erased_samples)


    for epoch in range(epochs):
        similarity_term = []
        loss_list = []
        for grad, img in train_loader:
            grad, img = grad.to(args.device), img.to(args.device)  # (B, C, H, W), (B, 10)
            grad = grad.view(grad.size(0), 1, 16, 16)
            # img = img.view(img.size(0), -1)  # Flatten the images
            output = reconstructor(grad)
            # output = output.view(output.size(0), 3, 32, 32)
            x_hat = vib.reconstruction(output, img)
            img = img.view(img.size(0), -1)  # Flatten the images
            loss = reconstruction_function(x_hat, img)

            optimizer_recon.zero_grad()
            optimizer_vib_reconstruction.zero_grad()
            loss.backward()
            optimizer_recon.step()
            optimizer_vib_reconstruction.step()
            cos_sim = cosine_similarity(x_hat.view(1, -1), img.view(1, -1))
            similarity_term.append(cos_sim.item())
            loss_list.append(loss.item())



        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        print("cosine similarity:", sum(similarity_term)/len(similarity_term), "average loss:", sum(loss_list)/len(loss_list))
    return reconstructor


def evaluate_reconstructor(vib, reconstructor, train_loader, reconstruction_function, args):
    # init reconsturctor
    # reconstructor = LinearModel(n_feature=40, n_output=28 * 28).to(args.device)
    vib.decoder.trainable = False
    vib.fc3.trainable = False


    optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)
    reconstructor.train()
    epochs = 1

    ## training epochs
    total_training_samples = 50000*0.06
    erased_samples = 50000*args.erased_local_r
    epochs= int(total_training_samples/erased_samples)


    for epoch in range(1):
        similarity_term = []
        loss_list = []
        for grad, img in train_loader:
            grad, img = grad.to(args.device), img.to(args.device)  # (B, C, H, W), (B, 10)
            grad = grad.view(grad.size(0), 1, 16, 16)
            # img = img.view(img.size(0), -1)  # Flatten the images
            output = reconstructor(grad)
            # output = output.view(output.size(0), 3, 32, 32)
            x_hat = vib.reconstruction(output, img)
            img = img.view(img.size(0), -1)  # Flatten the images
            loss = reconstruction_function(x_hat, img)

            # optimizer_recon.zero_grad()
            # loss.backward()
            # optimizer_recon.step()
            cos_sim = cosine_similarity(x_hat.view(1, -1), img.view(1, -1))
            similarity_term.append(cos_sim.item())
            loss_list.append(loss.item())
            break



        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        print("constructed learning cosine similarity:", sum(similarity_term)/len(similarity_term), "average loss:", sum(loss_list)/len(loss_list))
    return reconstructor

# Function to add Laplace noise
def add_laplace_noise(tensor, epsilon, sensitivity, args):
    """
    Adds Laplace noise to a tensor for differential privacy.

    :param tensor: Input tensor
    :param epsilon: Privacy budget
    :param sensitivity: Sensitivity of the query/function
    :return: Noisy tensor
    """
    # Compute the scale of the Laplace distribution
    scale = sensitivity / epsilon

    # Generate Laplace noise
    noise = torch.tensor(np.random.laplace(0, scale, tensor.shape), dtype=tensor.dtype).to(args.device)

    # Add noise to the original tensor
    noisy_tensor = tensor + noise

    return noisy_tensor



seed = 0
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.use_deterministic_algorithms(True)

# parse args
args = args_parser()
args.gpu = 0
# args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.model = 'z_linear'
args.num_epochs = 1
args.dataset = 'CelebA'
args.add_noise = False
args.beta = 0.0001
args.mse_rate = 10
args.lr = 0.0001
args.unlearn_learning_rate = 0.1
args.ep_distance = 20
args.dimZ = 128  # 40 # 2
args.batch_size = 160
args.erased_local_r = 0.006 #0.000009 #0.006  # when ratio value is 0.000009, the sample is 1. 0.002 # the erased data ratio
args.construct_size = 0.006
args.auxiliary_size = 0.006
args.train_type = "MULTI"
args.kld_to_org = 1
args.unlearn_bce = 0.3
args.self_sharing_rate = 0.8
args.laplace_scale = 1
args.laplace_epsilon = 10
args.num_epochs_recon = 50

### 1: 0.000017, 20: 0.00034, 40: 0.00067, 60: 0.001, 80: 0.00134, 100: 0.00167
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
    train_set = MNIST('../data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('../data/mnist', train=False, transform=trans_mnist, download=True)
    train_set_no_aug = train_set
elif args.dataset == 'CIFAR10':
    train_transform = T.Compose([  # T.RandomCrop(32, padding=4),
        T.ToTensor(),
    ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
    test_transform = T.Compose([T.ToTensor(),
                                ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    train_set = CIFAR10('../data/cifar', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('../data/cifar', train=False, transform=test_transform, download=True)
    train_set_no_aug = CIFAR10('../data/cifar', train=True, transform=test_transform, download=True)
elif args.dataset == 'CelebA':
    train_transform = T.Compose([T.Resize((32, 32)),
                                 T.ToTensor(),
                                 ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
    test_transform = T.Compose([T.ToTensor(),
                                ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    #/kaggle/input/celeba/
    data_path = '../data/CelebA'
    train_set = CelebA(data_path, split='train', target_type = 'attr', transform=train_transform, download=False)
    test_set = CelebA(data_path, split='test', target_type = 'attr', transform=train_transform, download=False)



train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

full_size = len(train_set)

# the erasing data
erasing_size = int(full_size * args.erased_local_r)
remaining_size = full_size - erasing_size
remaining_set, erasing_set = torch.utils.data.random_split(train_set,
                                                           [remaining_size, erasing_size])

# the construction data size
construct_size = int(full_size * args.construct_size)
remaining_set_w_o_cons_size = remaining_size - construct_size
remaining_set_w_o_cons, construct_set = torch.utils.data.random_split(remaining_set,
                                                                      [remaining_set_w_o_cons_size, construct_size])

auxiliary_size = int(full_size * args.auxiliary_size)
remaining_set_wo_aux_size = remaining_set_w_o_cons_size - auxiliary_size
remaining_set_wo_aux, auxiliary_set = torch.utils.data.random_split(remaining_set_w_o_cons,
                                                                    [remaining_set_wo_aux_size, auxiliary_size])
# this is the final remaining dataset, without the erasing dataset, and the auxiliary dataset
print(len(remaining_set_wo_aux))
print('erased size',len(erasing_set))
# print(len(remaining_set_w_o_twin.dataset.data))

# it is the final remaining dataset
dataloader_remaining_after_aux = DataLoader(remaining_set_wo_aux, batch_size=args.batch_size, shuffle=True)

# it is the erasing dataset
dataloader_erasing = DataLoader(erasing_set, batch_size=args.batch_size, shuffle=True)

# it is the constructing dataset
dataloader_constructing = DataLoader(construct_set, batch_size=args.batch_size, shuffle=True)

# it is the auxiliary dataset
dataloader_auxiliary = DataLoader(auxiliary_set, batch_size=args.batch_size, shuffle=True)

dataset1 = dataloader_remaining_after_aux.dataset
dataset2 = dataloader_erasing.dataset
dataset3 = dataloader_constructing.dataset
dataset4 = dataloader_auxiliary.dataset

# Concatenate datasets and create a new loader, this is just an example to show how to concatenate the datasets
combined_dataset = ConcatDataset([dataset2, dataset3, dataset4])
combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

########################################################################
## determine target, we need to change the target data's label, we also need to keep the original label, do we need?


## add to one class
add_backdoor = 1  # =1 add backdoor , !=1 not add
# reshaped_data = data_reshape_to_np(erasing_subset, dataname=args.dataset)
# only set one sample as the watermark target, or multi samples?
# target_samples_size = erasing_size  # erasing_size  # len(erasing_subset.data)
mode = "Mark Erasing Data"
# feature_extra = SimpleCNN().cuda()

erasing_with_tri, erasing_with_tri_tar = add_trigger_new(add_backdoor, dataset2, erasing_size, mode)
# samples with backdoor trigger, not in the remainig dataset.
dataloader_erasing_with_tri = DataLoader(erasing_with_tri, batch_size=args.batch_size, shuffle=True)
dataloader_erasing_with_tri_tar = DataLoader(erasing_with_tri_tar, batch_size=args.batch_size, shuffle=True)

combined_dataset = ConcatDataset([dataset1, erasing_with_tri])
dataloader_remaining_and_erasing = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

sim_list = []
for step, (x, y) in enumerate(dataloader_erasing_with_tri):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    x_after_ldp = add_laplace_noise(x, epsilon=args.laplace_epsilon,sensitivity=1,args=args)
    x1_norm = F.normalize(x.view(-1), p=2, dim=0)
    x2_norm = F.normalize(x_after_ldp.view(-1), p=2, dim=0)

    # Calculate cosine similarity
    cos_similarity = torch.dot(x1_norm, x2_norm)
    sim_list.append(cos_similarity.item())

print("DP sim", sum(sim_list)/len(sim_list))



# poison_samples = int(full_size) * args.erased_local_r

x, y = erasing_with_tri[0]

# print(x)
x = x.cpu().data
x = x.clamp(0, 1)
if args.dataset == "MNIST":
    x = x.view(x.size(0), 1, 28, 28)
elif args.dataset == "CIFAR10":
    x = x.view(1, 3, 32, 32)
elif args.dataset == "CelebA":
    x = x.view(1, 3, 32, 32)

print(x)
grid = torchvision.utils.make_grid(x, nrow=1)
# grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
# plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
# plt.show()


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

# train VIB
clean_acc_list = []
mse_list = []

train_type = args.train_type

# we first train the model with the erasing data, and then we will unlearn it
start_time = time.time()
for epoch in range(args.num_epochs):
    vib.train()
    vib = vib_train(dataloader_remaining_and_erasing, vib, loss_fn, reconstruction_function, args, epoch,
                    train_type)  # dataloader_total, dataloader_w_o_twin

print('acc list', clean_acc_list)
print('mse list', mse_list)
end_time = time.time()
running_time = end_time - start_time
print(f'VIB Training took {running_time} seconds')

########


# dataloader_target_with_trigger = DataLoader(target_with_tri, batch_size=args.batch_size, shuffle=True)
vib.eval()
acc_r = eva_vib(vib, dataloader_remaining_after_aux, args, name='on clean remaining dataset', epoch=999)
backdoor_acc = eva_vib(vib, dataloader_erasing_with_tri, args, name='on backdoored erasing dataset', epoch=999)
acc = eva_vib(vib, test_loader, args, name='on test dataset', epoch=999)

# calculate unlearning difference

print("prepare unlearning update gradient")
unlearned_vib = copy.deepcopy(vib)
start_time = time.time()
unlearned_vib = prepare_unl(dataloader_erasing_with_tri, unlearned_vib, loss_fn, args, "no noise")
end_time = time.time()
running_time = end_time - start_time
print(f'unlearning {running_time} seconds')

unlearned_vib.eval()
acc_r = eva_vib(unlearned_vib, dataloader_remaining_after_aux, args, name='unlearned model on clean remaining dataset', epoch=999)
backdoor_acc = eva_vib(unlearned_vib, dataloader_erasing_with_tri, args, name='unlearned model on erased data', epoch=999)
acc = eva_vib(unlearned_vib, test_loader, args, name='unlearned model on test dataset', epoch=999)

print("prepare unlearning with DP noise")
unlearned_vib_with_noise = copy.deepcopy(vib)
start_time = time.time()
unlearned_vib_with_noise = prepare_unl(dataloader_erasing_with_tri, unlearned_vib, loss_fn, args, "noise")
end_time = time.time()
running_time = end_time - start_time
print(f'unlearning with dp {running_time} seconds')
unlearned_vib_with_noise.eval()
acc_r = eva_vib(unlearned_vib_with_noise, dataloader_remaining_after_aux, args, name='unlearned model on clean remaining dataset', epoch=999)
backdoor_acc = eva_vib(unlearned_vib_with_noise, dataloader_erasing_with_tri, args, name='unlearned model on erased data', epoch=999)
acc = eva_vib(unlearned_vib_with_noise, test_loader, args, name='unlearned model on test dataset', epoch=999)

update_deltas_direction = []
#     for name, param in model.named_parameters():
#         print(name)
for param1, param2 in zip(unlearned_vib.approximator.parameters(), vib.approximator.parameters()):
    # Calculate the difference (delta) needed to update model1 towards model2
    if param1.grad is not None:
        delta = param2.data.view(-1) - param1.data.view(-1)
        grad_direction = torch.sign(delta)
        update_deltas_direction.append(grad_direction)

## reconstruction

dim_z = 256
temp_grad = torch.empty(0, dim_z).float().to(args.device)
temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
empty_tensor = torch.Tensor([])
for step, (x, y) in enumerate(dataloader_erasing_with_tri):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    for (name1, param1), (name2, param2) in zip(unlearned_vib.named_parameters(), vib.named_parameters()):
        # Calculate the difference (delta) needed to update model1 towards model2
        if name1 == 'encoder.linear_head.bias':

            temp_img = torch.cat([temp_img, x], dim=0)
            delta = param2.data.view(-1) - param1.data.view(-1)
            flat_delta = torch.flatten(delta)
            # empty_tensor = torch.cat((empty_tensor, flat_delta), dim=0)
            B, C, H, W = x.size()

            mean = 1  # Mean of the distribution
            std_dev = 0.4  # Standard deviation of the distribution

            # Generate Gaussian noise
            random_tensor = torch.randn(B, dim_z) * std_dev + mean
            random_tensor = random_tensor.cuda()
            # random_tensor = torch.rand(B, dim_z).cuda()
            scaled_random_tensor = random_tensor / random_tensor.sum() * B * dim_z

            for scale_v in scaled_random_tensor:
                flat_delta = flat_delta.view(1, 256)
                mean_t_grad = flat_delta.mean()
                std_t_grad = flat_delta.std()
                flat_delta = flat_delta * scale_v
                temp_grad = torch.cat([temp_grad, flat_delta], dim=0)



er_reconstruction_set = Data.TensorDataset(temp_grad, temp_img)
er_reconstruction_set_loader = DataLoader(er_reconstruction_set, batch_size=args.batch_size, shuffle=True)

start_time = time.time()

reconstructor_er_re = train_reconstructor(copy.deepcopy(vib), er_reconstruction_set_loader, reconstruction_function, args)
end_time = time.time()
running_time_recon = end_time - start_time
print(f'reconstruction Training took {running_time_recon} seconds')



# Update the twin/anti watermarked samples
dataloader_constructing1 = copy.deepcopy(dataloader_constructing)
start_time = time.time()
fixed_vib = copy.deepcopy(vib)
fixed_vib_2 = copy.deepcopy(vib)
for epoch in range(1):
    dataloader_constructing1, fixed_vib = construct_input(dataloader_constructing1, unlearned_vib,
                                                            fixed_vib,
                                                            loss_fn, args,
                                                            epoch)
    # vib.train()
    # vib = vib_train(dataloader_constructing1, vib, loss_fn, reconstruction_function, args, epoch,
    #                 train_type)
    # backdoor_acc = eva_vib(vib, dataloader_erasing_with_tri, args, name='on erased', epoch=999)
end_time = time.time()

running_time = end_time - start_time
print(f'Constructing data {running_time} seconds')

start_time = time.time()
for epoch in range(1): # args.num_epochs
    vib.train()
    vib = vib_train(dataloader_constructing1, vib, loss_fn, reconstruction_function, args, epoch,
                    train_type)
    backdoor_acc = eva_vib(vib, dataloader_erasing_with_tri, args, name='on erased', epoch=999)


end_time = time.time()

running_time = end_time - start_time
print(f'OUL unlearning {running_time} seconds')

# dataloader_target_with_trigger = DataLoader(target_with_tri, batch_size=args.batch_size, shuffle=True)

vib.eval()
acc_r = eva_vib(vib, dataloader_remaining_after_aux, args, name='on clean remaining dataset', epoch=999)
backdoor_acc = eva_vib(vib, dataloader_erasing_with_tri, args, name='on erased data', epoch=999)
acc = eva_vib(vib, test_loader, args, name='on test dataset', epoch=999)


##
unlearned_params = [torch.sign(param.data.view(-1)) for param in unlearned_vib.approximator.parameters() if param.grad is not None]
vib_params = [torch.sign(param.data.view(-1)) for param in vib.approximator.parameters() if param.grad is not None]

# params_model = [p.data.view(-1) for p in model.parameters()]
p1 = torch.cat(unlearned_params)
p2 = torch.cat(vib_params)
cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0))
similarity_loss = 1 - cos_sim  # Loss is lower when similarity is higher

print("forgeability_sim", cos_sim.item())


forgeability_of_model_diff = []
#     for name, param in model.named_parameters():
#         print(name)
for param1, param2 in zip(unlearned_vib.approximator.parameters(), vib.approximator.parameters()):
    # Calculate the difference (delta) needed to update model1 towards model2
    if param1.grad is not None:
        delta = param2.data.view(-1) - param1.data.view(-1)
        forgeability_of_model_diff.append(torch.norm(delta, p=2).item())
print("forgeability", sum(forgeability_of_model_diff)/len(forgeability_of_model_diff))

dim_z = 256
temp_grad = torch.empty(0, dim_z).float().to(args.device)
temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
empty_tensor = torch.Tensor([])
for step, (x, y) in enumerate(dataloader_erasing_with_tri):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    for (name1, param1), (name2, param2) in zip(vib.named_parameters(), fixed_vib_2.named_parameters()):
        # Calculate the difference (delta) needed to update model1 towards model2
        if name1 == 'encoder.linear_head.bias':

            temp_img = torch.cat([temp_img, x], dim=0)
            delta = param2.data.view(-1) - param1.data.view(-1)
            flat_delta = torch.flatten(delta)
            # empty_tensor = torch.cat((empty_tensor, flat_delta), dim=0)
            B, C, H, W = x.size()

            mean = 1  # Mean of the distribution
            std_dev = 0.4  # Standard deviation of the distribution

            # Generate Gaussian noise
            random_tensor = torch.randn(B, dim_z) * std_dev + mean
            random_tensor = random_tensor.cuda()
            # random_tensor = torch.rand(B, dim_z).cuda()
            scaled_random_tensor = random_tensor / random_tensor.sum() * B * dim_z

            for scale_v in scaled_random_tensor:
                flat_delta = flat_delta.view(1, 256)
                mean_t_grad = flat_delta.mean()
                std_t_grad = flat_delta.std()
                flat_delta = flat_delta * scale_v
                temp_grad = torch.cat([temp_grad, flat_delta], dim=0)

er_reconstruction_set = Data.TensorDataset(temp_grad, temp_img)
er_reconstruction_set_loader = DataLoader(er_reconstruction_set, batch_size=args.batch_size, shuffle=True)

start_time = time.time()

reconstructor_er_re = train_reconstructor(copy.deepcopy(fixed_vib_2), er_reconstruction_set_loader, reconstruction_function, args)
end_time = time.time()
running_time_recon = end_time - start_time
print(f'reconstruction with unlearning intentions Training took {running_time_recon} seconds')



# dataloader_constructing1

# reconstruction without the knowledge of unlearning

dim_z = 256
temp_grad = torch.empty(0, dim_z).float().to(args.device)
temp_img_cons = torch.empty(0, 3, 32, 32).float().to(args.device)

empty_tensor = torch.Tensor([])
for step, (x, y) in enumerate(dataloader_constructing1):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    for (name1, param1), (name2, param2) in zip(vib.named_parameters(), fixed_vib_2.named_parameters()):
        # Calculate the difference (delta) needed to update model1 towards model2
        if name1 == 'encoder.linear_head.bias':

            temp_img_cons = torch.cat([temp_img_cons, x], dim=0)
            delta = param2.data.view(-1) - param1.data.view(-1)
            flat_delta = torch.flatten(delta)
            # empty_tensor = torch.cat((empty_tensor, flat_delta), dim=0)
            B, C, H, W = x.size()

            mean = 1  # Mean of the distribution
            std_dev = 0.4  # Standard deviation of the distribution

            # Generate Gaussian noise
            random_tensor = torch.randn(B, dim_z) * std_dev + mean
            random_tensor = random_tensor.cuda()
            # random_tensor = torch.rand(B, dim_z).cuda()
            scaled_random_tensor = random_tensor / random_tensor.sum() * B * dim_z

            for scale_v in scaled_random_tensor:
                flat_delta = flat_delta.view(1, 256)
                mean_t_grad = flat_delta.mean()
                std_t_grad = flat_delta.std()
                flat_delta = flat_delta * scale_v
                temp_grad = torch.cat([temp_grad, flat_delta], dim=0)

recon_set_cons = Data.TensorDataset(temp_grad, temp_img_cons)
cons_set_loader = DataLoader(recon_set_cons, batch_size=args.batch_size, shuffle=True)

start_time = time.time()

reconstructor_er_re = train_reconstructor(copy.deepcopy(fixed_vib_2), cons_set_loader, reconstruction_function, args)
reconstructor_er_re = evaluate_reconstructor(copy.deepcopy(fixed_vib_2), reconstructor_er_re, er_reconstruction_set_loader, reconstruction_function, args)
end_time = time.time()
running_time_recon = end_time - start_time
print(f'reconstruction with unlearning intentions Training took {running_time_recon} seconds')

