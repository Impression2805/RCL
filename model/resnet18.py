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
        # self.adapter = nn.Linear(512, 512, bias=False, device=device)
        #  self.linear = nn.Linear(512*block.expansion, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.center = nn.Linear(num_classes, 512 * block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
        # y = self.fc(out)
        # if feature_output:
        #     return y, out
        # else:
        #     return y

        if feature_output:
            return out, out
        else:
            return out

    def mixup_process(self, x_in, x_out, alpha=10.0):
        if x_in.size()[0] != x_out.size()[0]:
            length = min(x_in.size()[0], x_out.size()[0])
            x_in = x_in[:length]
            x_out = x_out[:length]
        lam = np.random.beta(alpha, alpha)
        x_oe = lam * x_in + (1 - lam) * x_out
        return x_oe, lam

    # def forward_manifold(self, x, length_size=0):
    #     layer_mix = np.random.randint(0, 4)
    #     lam_return = 0
    #     if layer_mix == 0:
    #         x_in = x[:length_size]
    #         x_out = x[length_size:]
    #         x_oe, lam = self.mixup_process(x_in, x_out)
    #         lam_return = lam
    #         x = torch.cat([x_in, x_oe], dim=0)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = self.layer1(x)
    #     if layer_mix == 1:
    #         x_in = x[:length_size]
    #         x_out = x[length_size:]
    #         x_oe, lam = self.mixup_process(x_in, x_out)
    #         lam_return = lam
    #         x = torch.cat([x_in, x_oe], dim=0)
    #     x = self.layer2(x)
    #     if layer_mix == 2:
    #         x_in = x[:length_size]
    #         x_out = x[length_size:]
    #         x_oe, lam = self.mixup_process(x_in, x_out)
    #         lam_return = lam
    #         x = torch.cat([x_in, x_oe], dim=0)
    #     x = self.layer3(x)
    #     if layer_mix == 3:
    #         x_in = x[:length_size]
    #         x_out = x[length_size:]
    #         x_oe, lam = self.mixup_process(x_in, x_out)
    #         lam_return = lam
    #         x = torch.cat([x_in, x_oe], dim=0)
    #     x = self.layer4(x)
    #     dim = x.size()[-1]
    #     x = F.avg_pool2d(x, dim)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x, lam_return


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
