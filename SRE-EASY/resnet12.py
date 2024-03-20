from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out


class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet12, self).__init__()
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))
        self.layers = nn.Sequential(*layers)
        # self.linear = linear(10 * feature_maps, num_classes)
        # self.rotations = rotations
        # self.linear_rot = linear(10 * feature_maps, 4)
        # replace
        self.linear = linear(10 * feature_maps * 2, num_classes)
        # self.linear = linear(10 * feature_maps, num_classes)

        self.rotations = rotations
        self.linear_rot = linear(10 * feature_maps * 2, 4)
        # self.linear_rot = linear(10 * feature_maps, 4)

        self.mp = nn.MaxPool2d((2, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # add prior
        self.prior = Prior(10 * feature_maps)

    def forward(self, x, index_mixup=None, lam=-1, return_map=False):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope=0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        # context_prior_map=None
        features, context_prior_map = self.prior(features)
        out = self.linear(features)
        # return out
        if return_map:
            if self.rotations:
                out_rot = self.linear_rot(features)
                return (out, out_rot, context_prior_map), features
            return (out, context_prior_map), features
        else:
            if self.rotations:
                out_rot = self.linear_rot(features)
                return (out, out_rot), features
            return out, features


class Prior(nn.Module):
    def __init__(self, in_planes):
        super(Prior, self).__init__()
        self.reduce_dim = in_planes

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layernorm1 = nn.LayerNorm(self.reduce_dim, elementwise_affine=True)
        self.layernorm2 = nn.LayerNorm(self.reduce_dim, elementwise_affine=True)

        self.fc_q = nn.Linear(self.reduce_dim, self.reduce_dim)
        self.fc_k = nn.Linear(self.reduce_dim, self.reduce_dim)
        self.fc_v = nn.Linear(self.reduce_dim, self.reduce_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        Q, K, V = self.fc_q(x), self.fc_k(x), self.fc_v(x)
        context_prior_map = torch.sigmoid(torch.matmul(Q, K.transpose(0, 1)) / self.reduce_dim ** 0.5)
        intra_context = torch.matmul(self.softmax(context_prior_map), V)
        total_context = torch.cat([self.layernorm1(intra_context), self.layernorm2(x)], dim=1)
        return total_context, context_prior_map
