import sys

sys.argv = ['']
del sys

import os
import math
from collections import defaultdict
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import save_image
import torchvision
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import copy
import random


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


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


class Unet(nn.Module):
    def __init__(self, in_channels, down_features, num_classes, pooling=False):
        super().__init__()
        self.expand = conv_block(in_channels, down_features[0])

        self.pooling = pooling

        down_stride = 1 if pooling else 2
        self.downs = nn.ModuleList([
            conv_block(ins, outs, stride=down_stride) for ins, outs in zip(down_features, down_features[1:])])

        up_features = down_features[::-1]
        self.ups = nn.ModuleList([
            conv_block(ins + outs, outs) for ins, outs in zip(up_features, up_features[1:])])

        self.final_conv = nn.Conv2d(down_features[0], num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.expand(x)

        x_skips = []

        for down in self.downs:
            x_skips.append(x)
            x = down(x)
            if self.pooling:
                x = F.max_pool2d(x, 2)

        for up, x_skip in zip(self.ups, reversed(x_skips)):
            x = torch.cat([self.upsample(x), x_skip], dim=1)
            x = up(x)

        x = self.final_conv(x)

        return x


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

class LinearModel_first_relu(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class My_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.data, self.targets = self.get_image_label()

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)

    def get_image_label(self, ):
        if args.dataset == "MNIST":
            temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.indices:
                image, label = self.dataset[id]
                image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
                # print(image)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
        elif args.dataset == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.indices:
                image, label = self.dataset[id]
                image, label = image.to(args.device).reshape(1, 3, 32, 32), torch.tensor([label]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

        print(temp_label.shape, temp_img.shape)
        d = Data.TensorDataset(temp_img, temp_label)
        return temp_img, temp_label


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.index = 0
        self.dataset = dataset
        #         self.idxs = list(idxs)
        #         self.idxs = random.sample(list(idxs), int(len(idxs)*sampling))
        if args.sampling == 1:
            self.idxs = list(idxs)
        else:
            self.idxs = np.random.choice(list(idxs), size=int(len(idxs) * args.sampling), replace=True)
            # self.idxs = random.sample(list(idxs), int(len(idxs) * sampling)) # without replacement
            # random.choice is with replacement
        # print('datasplite' , idxs, len(dataset))

        self.data, self.targets = self.get_image_label()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # print("item", item, self.index, self.idxs[item],label)
        self.index += 1
        # print("self.idxs", self.idxs)
        return image, label

    def get_image_label(self, ):
        if args.dataset == "MNIST":
            temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.idxs:
                image, label = self.dataset[id]
                image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
                # print(image)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
        elif args.dataset == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.idxs:
                image, label = self.dataset[id]
                image, label = image.to(args.device).reshape(1, 3, 32, 32), torch.tensor([label]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

        print(temp_label.shape, temp_img.shape)
        d = Data.TensorDataset(temp_img, temp_label)
        return temp_img, temp_label


class AdaHessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"

    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.requires_grad = True
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        # for p in params:
        #     # p.grad.requires_grad=True
        #     print(p.shape)
        #     print(p.grad.shape)
        grads = [p.grad for p in params]
        # grads.requires_grad = True
        # print("grads", grads)

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                # print(h_z, z)
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)
                # print("p.hess", p.hess.shape)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                #p.addcdiv_(exp_avg, denom, value=-step_size)
                p = p.addcdiv_(exp_avg, denom, value=step_size)

        return self.get_params()


    @staticmethod
    @torch.no_grad()
    def hessian_unl_update(p, hess, args, i):
        average_conv_kernel = False
        weight_decay = 0.0
        betas = (0.9, 0.999)
        hessian_power = 1.0
        eps = args.lr # 1e-8

        if average_conv_kernel and p.dim() == 4:
            hess = torch.abs(hess).mean(dim=[2, 3], keepdim=True).expand_as(hess).clone()

        # Perform correct stepweight decay as in AdamW
        # p = p.mul_(1 - args.lr * weight_decay)

        state = {}
        state["hessian"] = 1

        # State initialization
        if len(state) == 1:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
            state['exp_hessian_diag_sq'] = torch.zeros_like(
                p.data)  # Exponential moving average of Hessian diagonal square values

        exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
        beta1, beta2 = betas
        state['step'] = i

        # Decay the first and second moment running average coefficient

        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        #exp_hessian_diag_sq.mul_(beta2).addcmul_(p_hs.hess, p_hs.hess, value=1 - beta2)
        exp_hessian_diag_sq.mul_(beta2).addcmul_(hess, hess, value=1 - beta2)

        bias_correction1 = 1 #- beta1 ** state['step']
        bias_correction2 = 1 #- beta2 ** state['step']

        k = hessian_power
        denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(eps)

        # make update
        step_size = args.lr / bias_correction1
        # p.addcdiv_(exp_avg, denom, value=-step_size)
        p = p.addcdiv_(exp_avg, denom, value=step_size * 0.1)
        #p_hs.data = p_hs.data + args.lr * p_hs.grad.data * 10
        return exp_avg, denom, step_size



def scoring_function(matrix):
    # This is a simple scoring function that returns the matrix itself as scores.
    # You can replace it with your own scoring function if needed.
    return matrix

def dp_sampling(matrix, epsilon, sample_size, replacement):
    scores = scoring_function(matrix)
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()

    # Flatten the matrix and probabilities for sampling
    flat_matrix = matrix.flatten()
    flat_probabilities = probabilities.flatten()

    # Sample elements without replacement
    sampled_indices = np.random.choice(
        np.arange(len(flat_matrix)),
        size=sample_size,
        replace=replacement,
        p=flat_probabilities
    )

    # Create the output matrix with 0s
    output_matrix = np.zeros_like(matrix)

    # Set the sampled elements to their original values
    np.put(output_matrix, sampled_indices, flat_matrix[sampled_indices])

    return output_matrix
# matrix = torch.tensor([[2, 2, 2, 2], [2, 2, 2, 0]], dtype=torch.float32)
# epsilon = 1.0
# sample_size = 3
#
# sampled_matrix = dp_sampling(matrix, epsilon, sample_size)
# print(sampled_matrix)

class PoisonedDataset(Dataset):

    def __init__(self, dataset, base_label, trigger_label, poison_samples, mode="train", device=torch.device("cuda"),
                 dataname="MNIST", args =None, add_backdoor=1, dp_sample=0):
        # self.class_num = len(dataset.classes)
        # self.classes = dataset.classes
        # self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.add_backdoor = add_backdoor
        self.dp_sample = dp_sample
        self.args = args
        self.data, self.targets = self.add_trigger(self.reshape(dataset, dataname), dataset.targets, base_label,
                                                   trigger_label, poison_samples, mode)
        self.channels, self.width, self.height = self.__shape_info__()
        # self.data_test, self.targets_test = self.add_trigger_test(self.reshape(dataset.data, dataname), dataset.targets, base_label, trigger_label, portion, mode)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, dataset, dataname="MNIST"):
        if dataname == "MNIST":
            temp_img = dataset.data.reshape(len(dataset.data), 1, 28, 28).float()
        elif dataname == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().cuda()
            temp_label = torch.empty(0).long().cuda()
            for id in range(len(dataset)):
                image, label = dataset[id]
                image, label = image.cuda().reshape(1, 3, 32, 32), torch.tensor([label]).long().cuda()
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
                # print(id)

        # x = torch.Tensor(image.cuda())
        # x = torch.tensor(image)
        # # print(x)

        return np.array(temp_img.to("cpu"))

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, base_label, trigger_label, poison_samples, mode):
        print("## generate——test " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = []
        new_data_re = []

        # total_poison_num = int(len(new_data) * portion/10)
        _, width, height = data.shape[1:]
        for i in range(len(data)):
            if targets[i] == base_label:
                new_targets.append(trigger_label)
                if trigger_label != base_label:
                    if self.add_backdoor == 1:
                        new_data[i, :, width - 3, height - 3] = 255
                        new_data[i, :, width - 3, height - 4] = 255
                        new_data[i, :, width - 4, height - 3] = 255
                        new_data[i, :, width - 4, height - 4] = 255
                    # new_data[i, :, width - 23, height - 21] = 254
                    # new_data[i, :, width - 23, height - 22] = 254
                # new_data[i, :, width - 22, height - 21] = 254
                # new_data[i, :, width - 24, height - 21] = 254
                new_data[i] = new_data[i] / 255

                if self.dp_sample == 1:
                    replacement = False
                    sampled_matrix = dp_sampling(new_data[i], args.epsilon, args.dp_sampling_size, replacement)
                    new_data_re.append(sampled_matrix)
                elif self.dp_sample==2:
                    replacement = True
                    sampled_matrix = dp_sampling(new_data[i], args.epsilon, args.dp_sampling_size, replacement)
                    new_data_re.append(sampled_matrix)
                else:
                    new_data_re.append(new_data[i])
                # print("new_data[i]",new_data[i])
                poison_samples = poison_samples - 1
                if poison_samples <= 0:
                    break
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 1, 28, 28)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

        return torch.Tensor(new_data_re), torch.Tensor(new_targets).long()


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


# train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)

# train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)


def show_cifar(x):
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset == "MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset == "CIFAR10":
        x = x.view(1, 3, 32, 32)
    print(x)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()

def create_backdoor_train_dataset(dataname, train_data, base_label, trigger_label, poison_samples, batch_size, args , add_backdoor, dp_sample):
    train_data = PoisonedDataset(train_data, base_label, trigger_label, poison_samples=poison_samples, mode="train",
                                 device=args.device, dataname=dataname, args=args, add_backdoor=add_backdoor, dp_sample=dp_sample)
    b = Data.TensorDataset(train_data.data, train_data.targets)
    # x = test_data_tri.data_test[0]
    x = torch.tensor(train_data.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset == "MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset == "CIFAR10":
        x = x.view(1, 3, 32, 32)
    print(x)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return train_data.data, train_data.targets


"""
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 3, 32, 32)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()
"""


def create_backdoor_test_dataset(dataname, test_data, base_label, trigger_label, poison_samples, batch_size, device):
    test_data_tri = PoisonedDataset(test_data, base_label, trigger_label, poison_samples=poison_samples, mode="test",
                                    device=device, dataname=dataname)
    b = Data.TensorDataset(test_data_tri.data, test_data_tri.targets)
    # x = test_data_tri.data_test[0]
    x = torch.tensor(test_data_tri.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset == "MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset == "CIFAR10":
        x = x.view(1, 3, 32, 32)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return b


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



@torch.no_grad()
def test_category_acc(model, data_loader, args):

    model.eval()
    # Initialize counters
    correct_by_category = [0 for _ in range(10)]
    total_by_category = [0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            if args.dataset == 'MNIST':
                images = images.view(images.size(0), -1)
            outputs = model(images, mode='test')
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()

                if true_label == predicted_label:
                    correct_by_category[true_label] += 1
                total_by_category[true_label] += 1

    accuracy_by_category = [correct / total for correct, total in zip(correct_by_category, total_by_category)]

    for i, accuracy in enumerate(accuracy_by_category):
        print(f"Category {i}: {accuracy:.2%}")

def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def sample_gumbel(size):
    return -torch.log(-torch.log(torch.rand(size)))


def gumbel_reparametrize(log_p, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (B, num_samples, C)
    g = sample_gumbel(shape).to(log_p.device)  # (B, N, C)
    return F.softmax((log_p.unsqueeze(1) + g) / temp, dim=-1)  # (B, N, C)


# this is only a, at most k-hot relaxation
def k_hot_relaxed(log_p, k, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (k, B, C)
    k_log_p = log_p.unsqueeze(0).expand(shape).reshape((k * B, C))  # (k* B, C)
    k_hot = gumbel_reparametrize(k_log_p, temp, num_samples)  # (k* B, N, C)
    k_hot = k_hot.reshape((k, B, num_samples, C))  # (k, B, N, C)
    k_hot, _ = k_hot.max(dim=0)  # (B, N, C)
    return k_hot  # (B, N, C)


# needed for when labels are not one-hot
def soft_cross_entropy_loss(logits, y):
    return -(y * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


def KL_between_q_p(q_distr, p_distr):
    return 1


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


class MINE(nn.Module):
    def __init__(self, output_size=49, input_size=28*28, hidden_size=1280):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(input_size + output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        xy = torch.cat([x, y], dim=1)
        h = torch.relu(self.fc1(xy))
        return self.fc2(h)

class VIBI(nn.Module):
    def __init__(self, explainer, approximator, forgetter, k=4, num_samples=4, temp=1):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator
        self.forgetter = forgetter
        # self.fc3 = nn.Linear(49, 400)
        # self.fc4 = nn.Linear(400, 784)
        self.k = k
        self.temp = temp
        self.num_samples = num_samples

        self.warmup = False

    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """
        double_logits_z = self.explainer(x)  # (B, C, h, w)
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
        elif mode == 'forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'cifar_forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #print("logits_z, mu, logvar", logits_z, mu, logvar)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.cifar_forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'cifar_test':
            logits_z = self.explain(x, mode='test')  # (B, C, H, W)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)
            return logits_y
        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y
        elif mode == 'forgetting_from_Z':
            #logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_z = x
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.forget(logits_z)
            return logits_z, logits_y, x_hat #, mu, logvar

    def forget(self, logits_z):
        output_x = self.forgetter(logits_z)
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


def init_vibi(dataset):
    k = args.k
    beta = args.beta
    num_samples = args.num_samples
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if dataset == 'MNIST':
        approximator = LinearModel(n_feature=49)
        forgetter = LinearModel(n_feature=49, n_output= 28 * 28)
        explainer = LinearModel(n_feature=28 * 28, n_output=49 * 2)  # resnet18(1, 49*2) #
        lr = args.lr

    elif dataset == 'CIFAR10':
        approximator = LinearModel(n_feature=3 * 7 * 7)
        explainer = resnet18(3, 3 * 7 * 7 * 2)  # resnet18(1, 49*2)
        forgetter = LinearModel(n_feature=3 * 7 * 7, n_output=3 * 32 * 32)
        lr = args.lr

    elif dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=8 * 8 * 8, n_output=100)
        explainer = resnet18(3, 8 * 8 * 8 * 2)  # resnet18(1, 49*2)
        forgetter = LinearModel(n_feature=8 * 8 * 8, n_output=3 * 32 * 32)
        lr = 3e-4
    elif dataset == 'MNIST_nips':
        approximator = LinearModel_first_relu(n_feature=49)
        forgetter = LinearModel(n_feature=49, n_output= 28 * 28)
        explainer = LinearModel(n_feature=28 * 28, n_output=49 * 2)  # resnet18(1, 49*2) #
        lr = args.lr

    vibi = VIBI(explainer, approximator, forgetter, k=k, num_samples=args.num_samples)
    vibi.to(args.device)
    return vibi, lr

def test_grad(model, x, y, loss_fn, optimizer):
    logits_z, logits_y, x_hat, mu, logvar = model(x.detach(), mode='forgetting')  # (B, C* h* w), (B, N, 10)
    H_p_q = loss_fn(logits_y, y)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

    x_hat = x_hat.view(x_hat.size(0), -1)
    x = x.view(x.size(0), -1)
    # x = torch.sigmoid(torch.relu(x))
    BCE = reconstruction_function(x_hat, x)  # mse loss
    loss1 = args.beta * KLD_mean
    loss2 = H_p_q
    optimizer.zero_grad()
    print("loss2.grad 1", loss2.grad)
    loss2.backward()

    print("loss2.grad2", loss2.grad)
    H_p_q.detach()
    optimizer.zero_grad()



def learning_train(dataset, model, loss_fn, reconstruction_function, args, epoch, mu_list,
                   sigma_list, train_loader, learn_model_type):
    logs = defaultdict(list)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        # print(x)
        # break

        #test_grad(model, x, y, loss_fn, optimizer)


        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        H_p_q = loss_fn(logits_y, y)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()


        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss

        if learn_model_type == 'vib':
            loss = args.beta * KLD_mean + H_p_q  # + BCE / (args.batch_size * 28 * 28)
        elif learn_model_type == 'nips':
            loss = 0.0000001 * KLD_mean + H_p_q

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        # B, Z_size = logits_z.shape
        # mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'loss': loss.item(),
            'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
            # 'mutual_info':mi,
        }

        for m, v in metrics.items():
            logs[m].append(v)
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 2600 == 0:  # former is 600 when batch_size = 100
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
    return model, mu_list, sigma_list


def unlearning_frkl(vibi_f_frkl, optimizer_frkl, vibi, epoch_test_acc, dataloader_erase, dataloader_remain, loss_fn,
                    reconstructor, reconstruction_function, test_loader, train_loader, train_type):
    logs = defaultdict(list)

    acc_test = []
    backdoor_acc_list = []

    print(len(dataloader_erase.dataset))
    train_bs = 0
    temp_acc = []
    temp_back = []
    for epoch in range(args.num_epochs):
        vibi_f_frkl.train()
        step_start = epoch * len(dataloader_erase)
        index = 0

        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x2 = x2.view(x2.size(0), -1)


            logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            logits_z_e2, logits_y_e2, x_hat_e2, mu_e2, logvar_e2 = vibi_f_frkl(x2, mode='forgetting')
            logits_z_f, logits_y_f, x_hat_f, mu_f, logvar_f = vibi(x, mode='forgetting')
            # logits_y_e = torch.softmax(logits_y_e, dim=1)
            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)

            H_p_q = loss_fn(logits_y_e, y)

            H_p_q2 = loss_fn(logits_y_e2, y2)
            KLD_element2 = mu_e2.pow(2).add_(logvar_e2.exp()).mul_(-1).add_(1).add_(logvar_e2).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            KLD = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KLD_mean = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KL_z_r = (torch.exp(logits_z_e_log_softmax) * logits_z_e_log_softmax).sum(dim=1).mean() + math.log(
                logits_z_e_log_softmax.shape[1])

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

            x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
            x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(x)
            BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            e_log_p = torch.exp(BCE / (args.batch_size * 28 * 28))  # = 1/p
            e_log_py = torch.exp(H_p_q)
            log_z = torch.mean(logits_z_e.log_softmax(dim=1))
            log_y = torch.mean(logits_y_e.log_softmax(dim=1))
            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            kl_f_e = kl_loss(F.log_softmax(logits_y_e, dim=1), F.log_softmax(logits_y_f, dim=1))
            # loss = args.beta * KLD_mean + H_p_q - BCE / (args.batch_size * 28 * 28) - log_z / e_log_p

            # loss = KLD_mean - BCE + args.unlearn_learning_rate * (
            #             kl_f_e - H_p_q)  # #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2

            #print(KLD_mean.item(), BCE.item(), kl_f_e.item(), H_p_q.item(), log_z.item(), log_y.item(), H_p_q2.item())


            unlearning_item = args.kld_r * KLD_mean.item() - args.unlearn_bce_r * BCE.item() + args.unlearn_ykl_r * kl_f_e.item() -  args.unlearn_learning_rate * H_p_q.item() - args.reverse_rate * (log_z.item() + log_y.item())

            #print(unlearning_item)
            learning_item = args.self_sharing_rate * (args.beta * KLD_mean2.item()+ H_p_q2.item())
            #print(learning_item)


            total = unlearning_item + learning_item # expected to equal to 0
            if unlearning_item <= 0:# have approixmate to the retrained distribution and no need to unlearn
                unl_rate = 0
            else:
                unl_rate = unlearning_item / total

            self_s_rate = 1 - unl_rate


            '''purpose is to make the unlearning item =0, and the learning item =0 '''

            if train_type == 'VIBU':
                loss = args.kld_r * KLD_mean - args.unlearn_bce_r * BCE + args.unlearn_ykl_r * kl_f_e - args.unlearn_learning_rate * H_p_q - args.reverse_rate * (log_z + log_y)
            elif train_type == 'VIBU-SS':
                loss = (args.kld_r * KLD_mean - args.unlearn_bce_r * BCE + args.unlearn_ykl_r * kl_f_e -  args.unlearn_learning_rate * H_p_q - args.reverse_rate * (log_z + log_y) ) * unl_rate + self_s_rate * args.self_sharing_rate * (
                                   args.beta * KLD_mean2 + H_p_q2)  # args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2
            elif train_type == 'NIPSU':
                loss = args.kld_r * KLD_mean + args.unl_r_for_bayesian * (- H_p_q) - args.reverse_rate * (log_z + log_y)


            optimizer_frkl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vibi_f_frkl.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_frkl.step()
            acc_back = (logits_y_e.argmax(dim=1) == y).float().mean().item()
            acc = (logits_y_e2.argmax(dim=1) == y2).float().mean().item()
            #test_acc = test_accuracy(vibi_f_frkl, test_loader, args, name='vibi valid top1')
            temp_acc.append(acc)
            temp_back.append(acc_back)
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            metrics = {
                'unlearning_item': unlearning_item,
                'learning_item': learning_item,
                'acc': acc,
                'loss': loss.item(),
                'BCE': BCE.item(),
                'H(p,q)': H_p_q.item(),
                'kl_f_e': kl_f_e.item(),
                'H_p_q2': H_p_q2.item(),
                # '1-JS(p,q)': JS_p_q,
                # 'mu_e': torch.mean(mu_e).item(),
                # 'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                'KLD': KLD.item(),
                'e_log_p': e_log_p.item(),
                'log_z': log_z.item(),
                'KLD_mean': KLD_mean.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu_e).item())
            #     sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if index % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{index % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            index = index + 1
            train_bs = train_bs + 1
            if acc_back < 0.05:
                break

        vibi_f_frkl.eval()
        valid_acc_old = 0.8
        valid_acc = test_accuracy(vibi_f_frkl, test_loader, args, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
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
        if backdoor_acc < 0.02:
            print()
            print("end unlearn, train_bs", train_bs)
            break

    print("temp_acc", temp_acc)
    print("temp_back", temp_back)

    return vibi_f_frkl, optimizer_frkl, epoch_test_acc




def unlearning_frkl_compressed(vibi_f_frkl, optimizer_frkl, vibi, epoch_test_acc, dataloader_erase, dataloader_remain, loss_fn,
                    reconstructor, reconstruction_function, test_loader, train_loader, train_type):
    logs = defaultdict(list)

    acc_test = []
    backdoor_acc_list = []

    print(len(dataloader_erase.dataset))
    train_bs = 0
    temp_acc = []
    temp_back = []

    for epoch in range(args.num_epochs):
        vibi_f_frkl.train()
        step_start = epoch * len(dataloader_erase)
        index = 0
        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
            vibi_f_frkl.train()
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x2 = x2.view(x2.size(0), -1)


            #logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            #logits_z_e2, logits_y_e2, x_hat_e2, mu_e2, logvar_e2 = vibi_f_frkl(x2, mode='forgetting')
            logits_z_f, logits_y_f, x_hat_f, mu_f, logvar_f = vibi(x, mode='forgetting')
            #logits_z_f2, logits_y_f2, x_hat_f2, mu_f2, logvar_f2 = vibi(x2, mode='forgetting')


            #here, the logvar is simga.pow(2)
            if train_type == 'NIPSU':
                logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
                KLD_mean = 0.5 * torch.mean(logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()
            else:
                logits_z_e, logits_y_e, x_hat_e  = vibi_f_frkl(logits_z_f, mode='forgetting_from_Z')
            #logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')
            #logits_z_e2, logits_y_e2, x_hat_e2  = vibi_f_frkl(logits_z_f2, mode='forgetting_from_Z')

            #x2 = torch.rand(len(x2), 28*28).to(args.device)

            logits_z_e2, logits_y_e2, x_hat_e2, mu_e2, logvar_e2  = vibi_f_frkl(x2, mode='forgetting')

            #two distributions, p(Z_e|X_e) and p(Z|X), apart from the q(Z)
            # logits_y_e = torch.softmax(logits_y_e, dim=1)
            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)

            H_p_q = loss_fn(logits_y_e, y)

            H_p_q2 = loss_fn(logits_y_e2, y2)


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

            M = Mine1(noise_size=49, sample_size=49)
            M.to(args.device)
            M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)

            B, Z_size = logits_z_e.shape
            for i in range(args.mi_epoch):
                mi = calculate_MI(logits_z_e2.detach(), logits_z_e.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
                if mi < 0:
                    i=i-1

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)

            x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
            x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(x)
            BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            e_log_p = torch.exp(BCE / (args.batch_size * 28 * 28))  # = 1/p
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
                'BCE': BCE.item(),
                'H(p,q)': H_p_q.item(),
                'kl_f_e': kl_f_e.item(),
                'H_p_q2': H_p_q2.item(),
                'mutual_info_z_and_ze': mutual_info_z_and_ze.item(),
                # 'mu_e': torch.mean(mu_e).item(),
                # 'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                'KLD_mean_z_and_z_e': KLD_mean_z_and_z_e.item(),
                'torch.mean( rho )':torch.mean( rho ).item(),
                'e_log_p': e_log_p.item(),
                'log_z': log_z.item(),
                'KLD_mean_f': KLD_mean_f.item(),
                'KLD_mean_e2':KLD_mean2.item(),
                'mutual_info':mi,
            }

            for m, v in metrics.items():
                logs[m].append(v)
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu_e).item())
            #     sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if index % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{index % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            index = index + 1
            train_bs = train_bs+1
            if acc_back < 0.02:
                break

        vibi_f_frkl.eval()
        valid_acc_old = 0.8
        valid_acc = test_accuracy(vibi_f_frkl, test_loader, args, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
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

    return vibi_f_frkl, optimizer_frkl, epoch_test_acc



'''we should prepare the compressed Z, and then use Z to be unlearned. 
not directly unlearning as normal'''
def unlearning_frkl_train(vibi, dataloader_erase, dataloader_remain, loss_fn, reconstructor, reconstruction_function,
                          test_loader, train_loader, train_type='VIBU'):
    vibi_f_frkl, lr = init_vibi(args.dataset)
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

        print('Unlearning VIBI KLD')
        print(f'{args.explainer_type:>10} explainer params:\t{num_params(vibi_f_frkl.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi_f_frkl.approximator).__name__:>10} approximator params:\t{num_params(vibi_f_frkl.approximator) / 1000:.2f} K')
        print(
            f'{type(vibi_f_frkl.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_f_frkl.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        epoch_test_acc = []
        mu_list = []
        sigma_list = []
        ''' we should fix the original model to fix the Z
        here, vibi_f_frkl is the unlearning model and vibi is the original model 
        so we can used the original model to create the fixed Z
        '''
        vibi_f_frkl, optimizer_frkl, epoch_test_acc = unlearning_frkl_compressed(vibi_f_frkl, optimizer_frkl, vibi,
                                                                      epoch_test_acc, dataloader_erase,
                                                                      dataloader_remain, loss_fn,
                                                                      reconstructor, reconstruction_function,
                                                                      test_loader, train_loader, train_type)

        final_round_mse = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi.train()
            step_start = epoch * len(dataloader_erase)
            for step, (x, y) in enumerate(dataloader_erase, start=step_start):
                x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
                x = x.view(x.size(0), -1)
                logits_z, logits_y, x_hat_e, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

                x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
                x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
                x = x.view(x.size(0), -1)
                # x = torch.sigmoid(torch.relu(x))
                BCE = reconstruction_function(x_hat_e, x)  # mse loss
                loss = BCE

                optimizer_recon_for_un.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer_recon_for_un.step()
                if epoch == args.num_epochs - 1:
                    final_round_mse.append(BCE.item())
                if step % len(train_loader) % 600 == 0:
                    print("loss", loss.item(), 'BCE', BCE.item())

        print("final epoch mse", np.mean(final_round_mse))

        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z, logits_y, x_hat_e, mu, logvar = vibi_f_frkl(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
            x_hat_e = torch.sigmoid(reconstructor_for_unlearning(logits_z))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            x = x.view(x.size(0), -1)
            break

        print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))
        print("frkld epoch_test_acc", epoch_test_acc)
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
    return vibi_f_frkl, optimizer_frkl


def calculate_hs_p(KLD_mean2, H_p_q2, optimizer_hessian, vibi_f_hessian):
    loss = args.beta * KLD_mean2 + H_p_q2  # + BCE / (args.local_bs * 28 * 28)
    optimizer_hessian.zero_grad()

    # loss.backward()
    # log_probs = net(images)
    # loss = self.loss_func(log_probs, labels)

    loss.backward(create_graph=True)

    optimizer_hs = AdaHessian(vibi_f_hessian.parameters())
    # optimizer_hs.get_params()
    optimizer_hs.zero_hessian()
    optimizer_hs.set_hessian()

    params_with_hs = optimizer_hs.get_params()
    # optimizer_hessian.step()
    optimizer_hessian.zero_grad()
    vibi_f_hessian.zero_grad()

    return params_with_hs

def unlearning_hessian_train(vibi, dataloader_erase, remaining_set, loss_fn, reconstructor, reconstruction_function,
                             test_loader, train_loader, train_type='VIBU'):
    vibi_f_hessian, lr = init_vibi(args.dataset)
    vibi_f_hessian.to(args.device)
    vibi_f_hessian.load_state_dict(vibi.state_dict())
    optimizer_hessian = torch.optim.Adam(vibi_f_hessian.parameters(), lr=lr)

    init_epoch = 0
    print("unlearning")
    epoch_loss = []
    acc_list = []
    acc_test = []
    backdoor_acc_list = []
    params_with_hs = None

    logs = defaultdict(list)

    reconstructor_for_unlearning = LinearModel(n_feature=49, n_output=28 * 28)
    reconstructor_for_unlearning = reconstructor_for_unlearning.to(args.device)
    optimizer_recon_for_un = torch.optim.Adam(reconstructor_for_unlearning.parameters(), lr=lr)

    dataloader_remain = DataLoader(remaining_set, batch_size=remaining_set.__len__(), shuffle=True)
    dataloader_remain2 = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True)

    train_bs = 0
    # prepare hessian
    index = 0
    for batch_idx, (images, labels) in enumerate(dataloader_remain):
        images, labels = images.to(args.device), labels.to(args.device)
        B, c, h, w = images.shape
        # print(B,h,w)
        if args.dataset == 'MNIST':
            images = images.reshape((B, -1))
        vibi_f_hessian.zero_grad()
        print('batch_idx', batch_idx)
        logits_z, logits_y, x_hat, mu, logvar = vibi_f_hessian(images, mode='forgetting')  # (B, C* h* w), (B, N, 10)
        H_p_q = loss_fn(logits_y, labels)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        loss = args.beta * KLD_mean + H_p_q  # + BCE / (args.local_bs * 28 * 28)
        optimizer_hessian.zero_grad()

        # loss.backward()
        # log_probs = net(images)
        # loss = self.loss_func(log_probs, labels)

        loss.backward(create_graph=True)

        optimizer_hs = AdaHessian(vibi_f_hessian.parameters())
        # optimizer_hs.get_params()
        optimizer_hs.zero_hessian()
        optimizer_hs.set_hessian()

        params_with_hs = optimizer_hs.get_params()

        optimizer_hessian.zero_grad()
        vibi_f_hessian.zero_grad()

    if init_epoch == 0 or args.resume_training:

        print('Unlearning VIBI KLD')
        print(f'{args.explainer_type:>10} explainer params:\t{num_params(vibi_f_hessian.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi_f_hessian.approximator).__name__:>10} approximator params:\t{num_params(vibi_f_hessian.approximator) / 1000:.2f} K')
        print(
            f'{type(vibi_f_hessian.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_f_hessian.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        epoch_test_acc = []
        mu_list = []
        sigma_list = []

        # unlearning
        convergence = 0
        temp_acc = []
        temp_back = []

        for iter in range(args.num_epochs):  # self.args.local_ep
            batch_loss = []
            # print(iter)

            for (images, labels), (images2, labels2) in zip(dataloader_erase, dataloader_remain2):
                # for batch_idx, (images, labels) in enumerate(self.erased_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                B, c, h, w = images.shape
                # print(B,h,w)
                if args.dataset == 'MNIST':
                    images = images.reshape((B, -1))

                images2, labels2 = images2.to(args.device), labels2.to(args.device)
                B, c, h, w = images2.shape
                if args.dataset == 'MNIST':
                    images2 = images2.reshape((B, -1))

                vibi_f_hessian.zero_grad()
                logits_z, logits_y, x_hat, mu, logvar = vibi_f_hessian(images,
                                                                       mode='forgetting')  # (B, C* h* w), (B, N, 10)
                logits_z2, logits_y2, x_hat2, mu2, logvar2 = vibi_f_hessian(images2, mode='forgetting')

                ##remaining dataset used to unlearn

                H_p_q = loss_fn(logits_y, labels)
                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
                KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
                KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

                H_p_q2 = loss_fn(logits_y2, labels2)
                KLD_element2 = mu2.pow(2).add_(logvar2.exp()).mul_(-1).add_(1).add_(logvar2).cuda()
                KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).cuda()

                params_with_hs = calculate_hs_p(KLD_mean2, H_p_q2, optimizer_hessian, vibi_f_hessian)



                loss = args.beta * KLD_mean + H_p_q  # + BCE / (args.local_bs * 28 * 28)
                optimizer_hessian.zero_grad()

                loss.backward()
                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)

                i = 0
                for p_hs, p in zip(params_with_hs, vibi_f_hessian.parameters()):
                    i = i + 1
                    # if i==1:
                    #     continue
                    # print(p_hs.hess)
                    # break
                    temp_hs = torch.tensor(p_hs.hess)
                    # temp_hs = temp_hs.__add__(args.lr)
                    # p.data = p.data.addcdiv_(exp_avg, denom, value=-step_size * 10000)

                    # print(p.data)
                    # p.data = p_hs.data.addcdiv_(exp_avg, denom, value=step_size * args.lr)
                    if p.grad != None:
                        exp_avg, denom, step_size = AdaHessian.hessian_unl_update(p, temp_hs, args, i)
                        # print(exp_avg)
                        # print(denom)
                        p.data = p.data.addcdiv_(exp_avg, denom, value=args.hessian_rate)
                        # p.data =p.data + torch.div(p.grad.data, temp_hs) * args.lr #torch.mul(p_hs.hess, p.grad)*10
                        # print(p.grad.data.shape)
                    else:
                        p.data = p.data

                vibi_f_hessian.zero_grad()

                fl_acc = (logits_y.argmax(dim=1) == labels).float().mean().item()
                fl_acc2 = (logits_y2.argmax(dim=1) == labels2).float().mean().item()
                temp_acc.append(fl_acc2)
                temp_back.append(fl_acc)
                train_bs = train_bs + 1
                if fl_acc < 0.05:
                    break

                batch_loss.append(loss.item())

            vibi_f_hessian.eval()
            valid_acc_old = 0.8
            valid_acc = test_accuracy(vibi_f_hessian, test_loader, args, name='vibi valid top1')
            interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            logs['val_acc'].extend(interpolate_valid_acc)
            print("test_acc", valid_acc)
            epoch_test_acc.append(valid_acc)
            # valid_acc_old = valid_acc
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            print("test_acc", valid_acc)
            acc_test.append(valid_acc)
            backdoor_acc = test_accuracy(vibi_f_hessian, dataloader_erase, args, name='vibi valid top1')
            backdoor_acc_list.append(backdoor_acc)
            print("backdoor_acc", backdoor_acc_list)
            print("acc_test: ", acc_test)

            if backdoor_acc < 0.1:
                print()
                print("end hessian unl", train_bs)
                break
        print("temp_acc", temp_acc)
        print("temp_back", temp_back)
    return vibi_f_hessian, optimizer_hessian


def retraining_train(vibi, vibi_retrain, vibi_f_nipsu, vibi_f_frkl, vibi_f_frkl_ss, dataloader_remain, dataloader_erase, reconstructor,
                     reconstruction_function,
                     loss_fn, optimizer_retrain, test_loader, train_loader):
    init_epoch = 0
    print("retraining")
    logs = defaultdict(list)
    valid_acc = 0.8
    acc_test = []
    poison_acc = []
    backdoor_acc_list = []

    if init_epoch == 0 or args.resume_training:

        print('Retraining VIBI')
        print(f'{args.explainer_type:>10} explainer params:\t{num_params(vibi_retrain.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi_retrain.approximator).__name__:>10} approximator params:\t{num_params(vibi_retrain.approximator) / 1000:.2f} K')
        print(
            f'{type(vibi_retrain.forgetter).__name__:>10} forgetter params:\t{num_params(vibi_retrain.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        epoch_test_acc = []
        KL_fr = []
        KL_er = []
        KL_hr = []
        KL_ssr = []
        KL_nipsr = []
        KL_kl = []
        mu_list_f = []
        sigma_list_f = []
        mu_list_e = []
        sigma_list_e = []
        mu_list_r = []
        sigma_list_r = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi_retrain.train()
            step_start = epoch * len(dataloader_remain)
            for step, (x, y) in enumerate(dataloader_remain, start=step_start):

                x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
                if args.dataset == 'MNIST':
                    x = x.view(x.size(0), -1)
                logits_z_r, logits_y_r, x_hat_r, mu_r, logvar_r = vibi_retrain(x,
                                                                               mode='forgetting')  # (B, C* h* w), (B, N, 10)

                # logits_z_e_h, logits_y_e_h, x_hat_e_h, mu_e_h, logvar_e_h = vibi_f_hessian(x, mode='forgetting')
                logits_z_e, logits_y_e, x_hat_e, mu_e, logvar_e = vibi_f_frkl(x, mode='forgetting')

                logits_z_e_nips, logits_y_e_nips, x_hat_e_nips, mu_e_nips, logvar_e_nips = vibi_f_nipsu(x, mode='forgetting')

                # logits_z_e_kl, logits_y_e_kl, x_hat_e_kl, mu_e_kl, logvar_e_kl = vibi_f_kl(x, mode='forgetting')
                logits_z_e_ss, logits_y_e_ss, x_hat_e_ss, mu_e_ss, logvar_e_ss = vibi_f_frkl_ss(x,
                                                                                                       mode='forgetting')
                # print(x_hat_e)
                logits_z_f, logits_y_f, mu_f, logvar_f = vibi(x, mode='distribution')
                # logits_y_r = torch.softmax(logits_y_r, dim=1)
                logits_z_r_softmax = logits_z_r.log_softmax(dim=1)
                p_x_r = x.softmax(dim=1)

                KLD_element = mu_r.pow(2).add_(logvar_r.exp()).mul_(-1).add_(1).add_(logvar_r).cuda()
                KLD = torch.mean(KLD_element).mul_(-0.5).cuda()
                KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
                # x_hat_r = torch.sigmoid(reconstructor(logits_z_r))
                x_hat_r = x_hat_r.view(x_hat_r.size(0), -1)
                x = x.view(x.size(0), -1)

                #x_hat_e_h = torch.sigmoid(reconstructor(logits_z_e_h))
                # x_hat_e_h = x_hat_e_h.view(x_hat_e_h.size(0), -1)

                x_hat_f = torch.sigmoid(reconstructor(logits_z_f))
                x_hat_f = x_hat_f.view(x_hat_f.size(0), -1)
                BCE = reconstruction_function(x_hat_r, x)  # mse loss
                H_p_q = loss_fn(logits_y_r, y)
                loss_r = args.beta * KLD_mean + H_p_q  # + BCE / (args.batch_size * 28 * 28)

                optimizer_retrain.zero_grad()
                loss_r.backward()
                torch.nn.utils.clip_grad_norm_(vibi_retrain.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer_retrain.step()

                KLD_fr = 0.5 * torch.mean(
                    logvar_r - logvar_f + (torch.exp(logvar_f) + (mu_f - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

                # KLD_hr = 0.5 * torch.mean(
                #     logvar_r - logvar_e_h + (torch.exp(logvar_e_h) + (mu_e_h - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

                KLD_er = 0.5 * torch.mean(
                    logvar_r - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

                KLD_nipsr = 0.5 * torch.mean(
                    logvar_r - logvar_e_nips + (torch.exp(logvar_e_nips) + (mu_e_nips - mu_r).pow(2)) / torch.exp(logvar_r) - 1)

                # KLD_klr = 0.5 * torch.mean(
                #     logvar_r - logvar_e_kl + (torch.exp(logvar_e_kl) + (mu_e_kl - mu_r).pow(2)) / torch.exp(
                #         logvar_r) - 1)
                KLD_ss = 0.5 * torch.mean(
                    logvar_r - logvar_e_ss + (torch.exp(logvar_e_ss) + (mu_e_ss - mu_r).pow(2)) / torch.exp(
                        logvar_r) - 1)
                acc = (logits_y_r.argmax(dim=1) == y).float().mean().item()
                # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
                metrics = {
                    'acc': acc,
                    'loss': loss_r.item(),
                    'BCE': BCE.item(),
                    'H(p,q)': H_p_q.item(),
                    'mu_r': torch.mean(mu_r).item(),
                    'sigma_r': torch.sqrt_(torch.exp(logvar_r)).mean().item(),
                    'KLD_fr': KLD_fr.item(),
                    # 'KLD_hr': KLD_hr.item(),
                    # 'KLD_er': KLD_er.item(),
                    # 'KLD_nipsr':KLD_nipsr.item(),
                    'KLD_mean': KLD_mean.item(),
                }

                for m, v in metrics.items():
                    logs[m].append(v)
                if epoch == args.num_epochs - 1:
                    KL_fr.append(KLD_fr.item())
                    # KL_hr.append(KLD_hr.item())
                    KL_er.append(KLD_er.item())
                    # # KL_kl.append(KLD_klr.item())
                    KL_nipsr.append(KLD_nipsr.item())
                    KL_ssr.append(KLD_ss.item())
                # if epoch == args.num_epochs - 1:
                #     mu_list_r.append(torch.mean(mu_r).item())
                #     sigma_list_r.append(torch.sqrt_(torch.exp(logvar_r)).mean().item())
                #     mu_list_f.append(torch.mean(mu_f).item())
                #     sigma_list_f.append(torch.sqrt_(torch.exp(logvar_f)).mean().item())
                #     mu_list_e.append(torch.mean(mu_e).item())
                #     sigma_list_e.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())

                if step % len(dataloader_remain) % 20000 == 0:
                    print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(dataloader_remain):3d}] '
                          + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

            vibi.eval()
            valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi_retrain, test_loader, args, name='vibi valid top1')
            interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            logs['val_acc'].extend(interpolate_valid_acc)
            print("test_acc", valid_acc)
            epoch_test_acc.append(valid_acc)

            print("epoch: ", epoch)
            # valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi_retrain, test_loader, args, name='vibi valid top1')
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            print("test_acc", valid_acc)
            acc_test.append(valid_acc)
            backdoor_acc = test_accuracy(vibi_retrain, dataloader_erase, args, name='vibi valid top1')
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            backdoor_acc_list.append(backdoor_acc)
            print("backdoor_acc", backdoor_acc_list)

        print('mu_r', np.mean(mu_list_r), 'sigma_r', np.mean(sigma_list_r))
        print('mu_e', np.mean(mu_list_e), 'sigma_e', np.mean(sigma_list_e))
        print('mu_f', np.mean(mu_list_f), 'sigma_f', np.mean(sigma_list_f))
        print("epoch_test_acc", epoch_test_acc)
        print("KL_fr", np.mean(KL_fr), "KL_hr", np.mean(KL_hr), "KL_nipsr", np.mean(KL_nipsr), "KL_er", np.mean(KL_er), "KL_ssr", np.mean(KL_ssr), "KL_kl", np.mean(KL_kl))
        # print(KL_er)

    return vibi_retrain



def learn_mine(G, M, M_opt,args , ma_rate=0.001):
    '''
    Mine is learning for MI of (input, output) of Generator.
    '''
    z = torch.randn((args.local_bs, 10))
    # which is the input of Generator.
    z_bar = torch.narrow(torch.randn((args.local_bs, 10)), dim=1, start=0, length=3)
    # which is for product distribution.
    if args.device:
        z = z.cuda()
        z_bar = z_bar.cuda()
    x_tilde = G(z)  # which is the output of Generator.

    et = torch.mean(torch.exp(M(z_bar, x_tilde)))
    if M.ma_et is None:
        M.ma_et = et.detach().item()
    M.ma_et += ma_rate * (et.detach().item() - M.ma_et)
    z = torch.narrow(z, dim=1, start=0, length=3)  # slice for MI
    mutual_information = torch.mean(M(z, x_tilde)) \
                         - torch.log(et) * et.detach() / M.ma_et
    loss = - mutual_information

    M_opt.zero_grad()
    loss.backward()
    M_opt.step()

    return mutual_information.item()


def calculate_MI(X, Z, Z_size, M, M_opt, args, ma_rate=0.001):
    '''
    we use Mine to calculate the mutual information between two layers of networks.
    :param G:
    :param M:
    :param ma_rate:
    :return:
    '''

    z_bar = torch.randn((args.local_bs, Z_size)).to(args.device)

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


def calcu_MI(X, Z, Z_size, M, M_opt, args, ma_rate=0.001):
    '''
    we use Mine to calculate the mutual information between two layers of networks.
    :param G:
    :param M:
    :param ma_rate:
    :return:
    '''

    z_bar = torch.randn((args.local_bs, Z_size)).to(args.device)

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



def print_multal_info(vibi_f_frkl_ss, dataloader_erase,dataloader_sampled, args):
    reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
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

    M = Mine1()
    M.to(args.device)
    M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)
    step_start=0
    mutual_training_round = int (0.1 / args.erased_local_r) + 1
    t_round =  int( len(dataloader_erase) / args.erased_local_r * 0.1)
    print(t_round, len(dataloader_erase))
    for i in range(mutual_training_round):
        #for step, (x, y) in enumerate(dataloader_erase, start=step_start):
        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_sampled):
            t_round = t_round - 1
            if t_round < 0 : break
            x, y = x.to(args.device), y.to(args.device)
            x = x.view(x.size(0), -1)
            x2, y2 = x2.to(args.device), y2.to(args.device)
            x2 = x2.view(x2.size(0), -1)
            logits_z, logits_y, x_hat, mu, logvar = vibi_f_frkl_ss(x2, mode='forgetting')  # (B, C* h* w), (B, N, 10)

            B, Z_size = logits_z.shape
            x = x.view(x.size(0), -1)
            mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
            if mi < 0:
                mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
            if t_round % 100 == 0:
                print(t_round, ' mutual info ', mi)

    print()
    print('mutual information after vibi_f_frkl_ss unlearning', mi)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
    print('kld_mean', KLD_mean.item())

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




def prepare_learning_model(dataloader_full, train_loader, test_loader, dataloader_erase,dataloader_remain,  logs, reconstructor,  reconstruction_function,optimizer_recon, explainer_type, init_epoch, loss_fn, args,valid_acc, learn_model_type):
    vibi, lr = init_vibi(args.dataset)
    vibi.to(args.device)

    M = Mine1()
    M.to(args.device)
    M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)

    for name, layer in vibi.explainer.named_children():
        print(name, layer)

    # fc2_weight = vibi.explainer.state_dict()['fc2.weight']
    # fc2_bias = vibi.explainer.state_dict()['fc2.bias']
    #
    # print(fc2_weight.shape)
    # print(fc2_bias.shape)

    acc_test = []
    print("learning")
    if init_epoch == 0 or args.resume_training:

        print('Training VIBI')
        print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
        print(
            f'{type(vibi.approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')
        print(f'{type(vibi.forgetter).__name__:>10} forgetter params:\t{num_params(vibi.forgetter) / 1000:.2f} K')
        # inspect_explanations()
        mu_list = []
        sigma_list = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi.train()
            step_start = epoch * len(dataloader_full)
            vibi, mu_list, sigma_list = learning_train(dataloader_full, vibi, loss_fn, reconstruction_function, args,
                                                       epoch, mu_list, sigma_list, train_loader, learn_model_type)
            vibi.eval()
            valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi, test_loader, args, name='vibi valid top1')
            interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            logs['val_acc'].extend(interpolate_valid_acc)
            print("test_acc", valid_acc)
            backdoor_acc = test_accuracy(vibi, dataloader_erase, args, name='vibi valid top1')
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            print("backdoor_acc", backdoor_acc)
            acc_test.append(valid_acc)

        print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))
        #
        final_round_mse = []
        for epoch in range(init_epoch, init_epoch + args.num_epochs):
            vibi.train()
            step_start = epoch * len(dataloader_erase)
            for step, (x, y) in enumerate(dataloader_erase, start=step_start):
                x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
                x = x.view(x.size(0), -1)
                logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

                x_hat = torch.sigmoid(reconstructor(logits_z))
                x_hat = x_hat.view(x_hat.size(0), -1)
                x = x.view(x.size(0), -1)
                # x = torch.sigmoid(torch.relu(x))
                BCE = reconstruction_function(x_hat, x)  # mse loss
                loss = BCE

                optimizer_recon.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
                optimizer_recon.step()

                if epoch == args.num_epochs - 1:
                    final_round_mse.append(BCE.item())
                if step % len(train_loader) % 600 == 0:
                    print("loss", loss.item(), 'BCE', BCE.item())

        print("final_round mse", np.mean(final_round_mse))

        for step, (x, y) in enumerate(test_loader, start=step_start):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            x = x.view(x.size(0), -1)
            logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)
            x_hat = torch.sigmoid(reconstructor(logits_z))
            x_hat = x_hat.view(x_hat.size(0), -1)
            x = x.view(x.size(0), -1)
            break

        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
            x, y = x.to(args.device), y.to(args.device)
            x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)
            x2 = x2.view(x2.size(0), -1)

            logits_z, logits_y, x_hat, mu, logvar = vibi(x2, mode='forgetting')  # (B, C* h* w), (B, N, 10)

            B, Z_size = logits_z.shape
            for i in range(10):
                mi = calculate_MI(x.detach(), logits_z.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
                if mi<0: i =i-1
                if i % 1 == 0:
                    print('mutual info', mi)
            print()
            print('mutual information after learning', mi)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
            KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
            print('kld_mean', KLD_mean.item())
            break

        x_hat_cpu = x_hat.cpu().data
        x_hat_cpu = x_hat_cpu.clamp(0, 1)
        x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()
        x_cpu = x.cpu().data
        x_cpu = x_cpu.clamp(0, 1)
        x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
        plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
        plt.show()
    return vibi



torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

#torch.use_deterministic_algorithms(True)

# parse args
args = args_parser()
args.gpu = 0
args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.model = 'z_linear'
args.local_bs = 100 #small batch_size has a better performance
args.local_ep = 10
args.num_epochs = 20
args.total_batch_round = 600
args.dataset = 'MNIST'
args.xpl_channels = 1
args.epochs = int(10)
args.add_noise = False
args.beta = 1 #0.001
args.lr = 0.001
args.max_norm=1
args.erased_size = 1500  # 120
args.poison_portion = 0.0
args.erased_portion = 0.3
args.erased_local_r = 0.06
args.batch_size = args.local_bs

args.epsilon = 1.0
args.dp_sampling_size = int(28*28*0.601)


## in unlearning, we should make the unlearned model first be backdoored and then forget the trigger effect
args.unlearn_learning_rate = 1
args.mi_rate = 4
args.mi_epoch=40
args.reverse_rate = 0.5
args.kld_r = 1
args.unlearn_ykl_r = args.unlearn_learning_rate*0.4
args.unlearn_bce_r = args.unlearn_learning_rate
args.unl_r_for_bayesian = args.unlearn_learning_rate
args.self_sharing_rate = args.unlearn_learning_rate*5   # compensation, small will perform better in erasure
args.unl_conver_r = 2
#args.hessian_rate = 0.005
args.hessian_rate = 0.00005
print('args.beta', args.beta, 'args.lr', args.lr)
print('args.erased_portion', args.erased_portion, 'args.erased_local_r', args.erased_local_r)
print('args.unlearn_learning_rate', args.unlearn_learning_rate, 'args.self_sharing_rate', args.self_sharing_rate)

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
    train_set = CIFAR10('../../data/cifar', train=True, transform=train_transform, download=False)
    test_set = CIFAR10('../../data/cifar', train=False, transform=test_transform, download=False)
    train_set_no_aug = CIFAR10('../../data/cifar', train=True, transform=test_transform, download=False)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

shadow_ratio = 0.0
full_ratio = 1 - shadow_ratio
unlearning_ratio = args.erased_local_r

length = len(train_set)
shadow_size, full_size = int(shadow_ratio * length), int(full_ratio * length)
remaining_size, erasing_size = int((1 - unlearning_ratio) * full_size), int(unlearning_ratio * full_size)
print('remaining_size', remaining_size)
remaining_size = full_size - erasing_size
print('remaining_size', remaining_size, shadow_size, full_size, erasing_size)

remaining_set, erasing_set = torch.utils.data.random_split(train_set, [remaining_size, erasing_size])

print(len(remaining_set))
print(len(remaining_set.dataset.data))

remaining_set = My_subset(remaining_set.dataset, remaining_set.indices)
clean_erasing_set = My_subset(erasing_set.dataset, erasing_set.indices)

# dataloader_shadow = DataLoader(shadow_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

poison_samples = int(length) * args.erased_local_r
base_label=3
trigger_label = 3
add_backdoor=0 # =1 add backdoor , !=1 not add
poison_data, poison_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.local_bs, args=args, add_backdoor=add_backdoor, dp_sample=0)


sampled_data, sample_labels = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.local_bs, args=args,  add_backdoor=add_backdoor,dp_sample=1) #dp_sample=1 without replacement


sampled_data_w, sample_labels_w = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.local_bs, args=args,  add_backdoor=add_backdoor,dp_sample=2) #dp_sample=2 with replacement


if args.dataset == 'MNIST':
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 1, 28, 28)
    #erasing_set.data = erasing_set.data.reshape(len(erasing_set.data), 1, 28, 28)
elif args.dataset == 'CIFAR10':
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 3, 32, 32)
    #erasing_set.data = erasing_set.data.reshape(len(erasing_set.data), 3, 32, 32)

print('train_set.data.shape', train_set.data.shape)
print('poison_data.shape', poison_data.shape)


data = torch.cat([poison_data.to(args.device), data_reshape.to(args.device)], dim=0)
targets = torch.cat([poison_targets.to(args.device), remaining_set.targets.to(args.device)], dim=0)

poison_trainset = Data.TensorDataset(data, targets)  # Data.TensorDataset(data, targets)
pure_backdoored_set = Data.TensorDataset(poison_data, poison_targets)

dp_sample_set = Data.TensorDataset(sampled_data, sample_labels)


dp_sample_set_w = Data.TensorDataset(sampled_data_w, sample_labels_w)

"""
we choose samples have the same label for unlearning the representer.
in a backdoored medol, we need to unlearn the trigger, 
so the remaining dataset is all the clean samples, and the erased dataset is the poisoned samples
here we set the pure_backdoored as the erased dataset
original erasing set is erasing_set = erasing_set"""
erasing_set = pure_backdoored_set
#remaining_set = same_label_set

#if we don't use poisoned set, we use full set
dataloader_full = DataLoader(poison_trainset, batch_size=args.batch_size, shuffle=True)

#dataloader_full = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
dataloader_remain = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True)
dataloader_clean_erase = DataLoader(clean_erasing_set, batch_size=args.batch_size, shuffle=True)
dataloader_dp_sampled = DataLoader(dp_sample_set, batch_size=args.batch_size, shuffle=False)
dataloader_dp_sampled_w = DataLoader(dp_sample_set_w, batch_size=args.batch_size, shuffle=False)
dataloader_erase = DataLoader(erasing_set, batch_size=args.batch_size, shuffle=True)

# for step, (x, y) in enumerate(dataloader_full):
#     print(x)
#     break
#
# for step, (x, y) in enumerate(dataloader_remain):
#     print(x)
#     break

#print('full size', len(poison_trainset), 'remain size', len(remaining_set.data), 'erased size', len(erasing_set))
print('full size', len(poison_trainset), 'remain size', len(remaining_set), 'erased size', len(erasing_set))

beta = args.beta

explainer_type = args.explainer_type

init_epoch = 0
best_acc = 0
logs = defaultdict(list)

reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
reconstructor = reconstructor.to(device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)

reconstruction_function = nn.MSELoss(size_average=False)

valid_acc = 0.8
loss_fn = nn.CrossEntropyLoss()


learn_model_type = 'vib'
vibi = prepare_learning_model(dataloader_full, train_loader, test_loader, dataloader_erase, dataloader_remain, logs, reconstructor, reconstruction_function, optimizer_recon, explainer_type, init_epoch, loss_fn, args, valid_acc,  learn_model_type)




# learn_model_type = 'nips'
# vibi_for_nips = prepare_learning_model(dataloader_full, train_loader, test_loader, dataloader_erase, dataloader_remain, logs, reconstructor, reconstruction_function, optimizer_recon, explainer_type, init_epoch, loss_fn, args,valid_acc,  learn_model_type)
#
#
#
#
# '''we should prepare the compressed Z, and then use Z to be unlearned.
# not directly unlearning as normal'''
#
#

# print()
# print("start hessian unlearning")
#
# vibi_f_hessian, optimizer_hessian = unlearning_hessian_train(copy.deepcopy(vibi).to(args.device), dataloader_erase, remaining_set, loss_fn,
#                                                              reconstructor, reconstruction_function,
#                                                              test_loader, train_loader, train_type='Hessian')
#
#
#
# print()
# print("start NIPSU")
#
# #dataloader_same_label
# vibi_f_nipsu, optimizer_nipsu = unlearning_frkl_train(copy.deepcopy(vibi).to(args.device), dataloader_erase, dataloader_remain, loss_fn,
#                                                     reconstructor,
#                                                     reconstruction_function, test_loader, train_loader,   train_type='NIPSU')

#




# print()
# print("start VIBU")
# vibi_f_frkl, optimizer_frkl = unlearning_frkl_train(copy.deepcopy(vibi).to(args.device), dataloader_erase, dataloader_remain, loss_fn,
#                                                     reconstructor,
#                                                     reconstruction_function, test_loader, train_loader,  train_type='VIBU')
#
#
# print_multal_info(vibi_f_frkl, dataloader_erase,dataloader_remain, args)


# print()
# print("start VIBU-SS")
# vibi_f_frkl_ss, optimizer_frkl_ss = unlearning_frkl_train(copy.deepcopy(vibi).to(args.device), dataloader_erase,
#                                                           dataloader_clean_erase, loss_fn,
#                                                           reconstructor,
#                                                           reconstruction_function, test_loader, train_loader, train_type='VIBU-SS')
#
#
# print_multal_info(vibi_f_frkl_ss, dataloader_erase,dataloader_remain, args)
#

#
# print()
# print("start VIBU-SS with dp sampling drop with replacement")
# #dataloader_dp_sampled
#
# vibi_f_frkl_sampled_w, optimizer_frkl_sampled_w = unlearning_frkl_train(copy.deepcopy(vibi).to(args.device), dataloader_dp_sampled_w,
#                                                           dataloader_clean_erase, loss_fn,
#                                                           reconstructor,
#                                                           reconstruction_function, test_loader, train_loader, train_type='VIBU-SS')
#
#
#
# # test_category_acc(vibi_f_frkl_sampled_w, test_loader, args)
#
# print()
# print("start VIBU-SS with dp sampling drop without replacement")
# #dataloader_dp_sampled
#
# vibi_f_frkl_sampled, optimizer_frkl_sampled = unlearning_frkl_train(copy.deepcopy(vibi).to(args.device), dataloader_dp_sampled,
#                                                           dataloader_clean_erase, loss_fn,
#                                                           reconstructor,
#                                                           reconstruction_function, test_loader, train_loader, train_type='VIBU-SS')
#

# print('mutual w info for vibi')
# print_multal_info(copy.deepcopy(vibi).to(args.device), dataloader_erase,dataloader_dp_sampled_w, args)
#
# print('mutual wo info for vibi')
# print_multal_info(copy.deepcopy(vibi).to(args.device), dataloader_erase,dataloader_dp_sampled, args)

# print('mutual info for vibi_for_nips')
# print_multal_info(vibi_for_nips, dataloader_erase,dataloader_erase, args)

# print('mutual vibi_f_hessian for vibi_for_nips')
# print_multal_info(vibi_f_hessian, dataloader_erase,dataloader_erase, args)
#
# print('mutual vibi_f_nipsu for vibi_for_nips')
# print_multal_info(vibi_f_nipsu, dataloader_erase,dataloader_erase, args)
#
# print('mutual vibi_f_frkl_sampled_w for vibi_for_nips')
# print_multal_info(vibi_f_frkl_sampled_w, dataloader_erase,dataloader_erase, args)
#
# print('mutual vibi_f_frkl_sampled for vibi_for_nips')
# print_multal_info(vibi_f_frkl_sampled, dataloader_erase,dataloader_erase, args)





'''

vibi_retrain, lr = init_vibi(args.dataset)
vibi_retrain.to(args.device)
optimizer_retrain = torch.optim.Adam(vibi_retrain.parameters(), lr=lr)


print()
print("start retraining")
vibi_retrain = retraining_train(vibi, vibi_retrain, vibi_f_nipsu, vibi_f_frkl, vibi_f_frkl_ss, dataloader_remain, dataloader_erase, reconstructor,
                                reconstruction_function,
                                loss_fn, optimizer_retrain, test_loader, train_loader)


print_multal_info(vibi_retrain, dataloader_erase,dataloader_remain, args)



reconstructor = LinearModel(n_feature=28*28, n_output=28 * 28)
reconstructor = reconstructor.to(device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=lr)

reconstruction_function = nn.MSELoss(size_average=False)

#
final_round_mse = []
for epoch in range(init_epoch, init_epoch + args.num_epochs):
    vibi.train()
    step_start = epoch * len(dataloader_erase)
    for step, (x, y) in enumerate(dataloader_erase, start=step_start):
        x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        logits_z, logits_y, x_hat, mu, logvar = vibi_retrain(x, mode='forgetting')  # (B, C* h* w), (B, N, 10)

        x_hat = torch.sigmoid(reconstructor(x))
        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss
        loss = BCE

        optimizer_recon.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vibi.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer_recon.step()

        if epoch == args.num_epochs - 1:
            final_round_mse.append(BCE.item())
        if step % len(train_loader) % 600 == 0:
            print("loss", loss.item(), 'BCE', BCE.item())

print("final_round mse retrain origin", np.mean(final_round_mse))




M = Mine1(noise_size=28*28, sample_size=28*28)
M.to(args.device)
M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)



for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
    x, y = x.to(device), y.to(device)
    x = x.view(x.size(0), -1)

    x2, y2 = x2.to(device), y2.to(device)
    x2 = x2.view(x2.size(0), -1)

    logits_z, logits_y, x_hat, mu, logvar = vibi_retrain(x2, mode='forgetting')  # (B, C* h* w), (B, N, 10)

    B, Z_size = x.shape

    for i in range(10):
        mi = calculate_MI(x.detach(), x.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
        if i % 1 == 0:
            print('mutual info', mi)
    print()
    print('mutual information after vibi_retrain unlearning', mi)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
    print('kld_mean', KLD_mean.item())
    break

print('Beta', beta)

'''