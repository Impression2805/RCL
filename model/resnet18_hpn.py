import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.mc_dropout = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 51300),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, ray=None, feature_output=False, embed=None):
        if embed is None:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            dim = out.size()[-1]
            out = F.avg_pool2d(out, dim)
            out = out.view(out.size(0), -1)
        else:
            out = embed
        if ray == None:
            # ray = torch.tensor([0.9, 0.01, 0.01])
            ray = torch.tensor([0.9, 0.1])
        fc_weight = self.ray_mlp(ray.to(torch.float32).cuda()).flatten()
        self.fc.weights = torch.nn.Parameter(fc_weight[0:51200].reshape(torch.Size([100, 512])))
        self.fc.bias = torch.nn.Parameter(fc_weight[51200:].reshape(torch.Size([100])))
        # self.fc.weights = torch.nn.Parameter(fc_weight[0:5120].reshape(torch.Size([10, 512])))
        # self.fc.bias = torch.nn.Parameter(fc_weight[5120:].reshape(torch.Size([10])))

        y = self.fc(out)
        return y
        # if feature_output:
        #     return y, out
        # else:
        #     return y
        # for _ in range(50):
        #     alpha = np.random.dirichlet(alpha=[0.1, 0.1], size=1)[0]
        #     # print(alpha)
        #     ray = torch.tensor([alpha[0], alpha[1]])
        #     fc_weight = self.ray_mlp(ray.to(torch.float32).cuda()).flatten()
        #     self.fc.weights = torch.nn.Parameter(fc_weight[0:51200].reshape(torch.Size([100, 512])))
        #     self.fc.bias = torch.nn.Parameter(fc_weight[51200:].reshape(torch.Size([100])))
        #     y += self.fc(out)
        #
        # return y/51


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
