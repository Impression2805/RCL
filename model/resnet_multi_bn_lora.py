import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.hub import load_state_dict_from_url
from pdb import set_trace
from .LoRA_for_resnet import Conv2d


def conv3x3(in_planes, out_planes, stride=1, r_nat=4):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, r_nat=r_nat)


def conv1x1(in_planes, out_planes, stride=1, r_nat=4):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, r_nat=r_nat)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, r_nat=4):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, r_nat=r_nat)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1, r_nat=r_nat)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # print(stride, in_planes, self.expansion*planes)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, stride=stride, r_nat=r_nat),
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )                    

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, r_nat=8)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, r_nat=8)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, r_nat=8)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, r_nat=8)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, r_nat=4)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, r_nat=4)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, r_nat=4)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, r_nat=4)

        self.fc = nn.Linear(512*block.expansion, num_classes)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, r_nat=4):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, r_nat=r_nat))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feature_output=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        dim = out.size()[-1]
        out = F.avg_pool2d(out, dim)
        out = out.view(out.size(0), -1)

        y = self.fc(out)
        if feature_output:
            return y, out
        else:
            return y



def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
