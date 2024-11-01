# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=100):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.mc_dropout = False
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         dim = out.size()[-1]
#         out = F.avg_pool2d(out, dim)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return y
#
#
# def ResNet18(num_classes):
#     return ResNet(BasicBlock, [2,2,2,2], num_classes)



import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class CosClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))

    def forward(self, x):
        score = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.mu, p=2, dim=1))
        score = score * 16
        return score

class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))  # each rotation have individual variance here
        self.temp = temp

    def forward(self, x, stochastic=True, test=False):
        mu = self.mu
        sigma = self.sigma
        sigma = F.softplus(sigma - 4)
        weight = sigma * torch.randn_like(mu) + mu
        # weight = F.normalize(weight, p=2, dim=1)
        # x = F.normalize(x, p=2, dim=1)
        score = F.linear(x, weight)
        if test:
            k = 100
            for _ in range(100):
                weight = sigma * torch.randn_like(mu) + mu
                weight = F.normalize(weight, p=2, dim=1)
                x = F.normalize(x, p=2, dim=1)
                score += F.linear(x, weight)
            score = score/(k+1)
        # score = score * 16
        return score


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.cosfc = CosClassifier(num_features=64 * block.expansion, num_classes=num_classes)
        # self.sfc = StochasticClassifier(num_features=64*block.expansion, num_classes=num_classes, temp=16)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, feature_output=False):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        last_feature = x.view(x.size(0), -1)

        # score = F.linear(F.normalize(last_feature, p=2, dim=1), F.normalize(self.fc.weight.data, p=2, dim=1))
        # x = score * 16
        x = self.fc(last_feature)
        if feature_output:
            return x, last_feature
        else:
            return x

        # if feature_output:
        #     return x, last_feature
        # else:
        #     return x

    def forward_pca(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # B, C, H, W = x.size()
        # x = x.view(B, C, H * W)
        # u, s, v = torch.linalg.svd(x, full_matrices=False)
        # for j in range(0, 1):
        #     x = x - s[:, j:(j+1)].unsqueeze(2) * u[:, :, j:(j+1)].bmm(v[:, j:(j+1), :])
        # if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        # feat2 = feat2 - power_iteration(feat2, iter=20)
        # x = x.view(B, C, H, W)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        last_feature = x.view(x.size(0), -1)
        u, s, v = torch.linalg.svd(last_feature, full_matrices=False)
        for j in range(10, 64):
            last_feature = last_feature - s[j] * u[:, j:(j+1)].mm(v[:, j:(j+1)].T)
        x = self.fc(last_feature)
        return x

    def forward_cos(self, x, feature_output=False):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        last_feature = x.view(x.size(0), -1)
        score = F.linear(F.normalize(last_feature, p=2, dim=1), F.normalize(self.fc.weight.data, p=2, dim=1))
        x = score * 1
        if feature_output:
            return x, last_feature
        else:
            return x

    def forward_threshold(self, x, threshold=1e10):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x_original = x
        x = x.clip(max=threshold)
        last_feature = x.view(x.size(0), -1)
        x = self.fc(last_feature)
        last_feature_original = x_original.view(x_original.size(0), -1)
        x_original = self.fc(last_feature_original)
        return x, x_original


    def mixup_process(self, x_in, x_out, alpha=10.0):
        if x_in.size()[0] != x_out.size()[0]:
            length = min(x_in.size()[0], x_out.size()[0])
            x_in = x_in[:length]
            x_out = x_out[:length]
        lam = np.random.beta(alpha, alpha)
        x_oe = lam * x_in + (1 - lam) * x_out
        return x_oe, lam

    def forward_manifold(self, x, length_size=0):
        layer_mix = np.random.randint(0, 3)
        lam_return = 0
        if layer_mix == 0:
            x_in = x[:length_size]
            x_out = x[length_size:]
            x_oe, lam = self.mixup_process(x_in, x_out)
            lam_return = lam
            x = torch.cat([x_in, x_oe], dim=0)
        x = self.conv1(x)
        x = self.layer1(x)
        if layer_mix == 1:
            x_in = x[:length_size]
            x_out = x[length_size:]
            x_oe, lam = self.mixup_process(x_in, x_out)
            lam_return = lam
            x = torch.cat([x_in, x_oe], dim=0)
        x = self.layer2(x)
        if layer_mix == 2:
            x_in = x[:length_size]
            x_out = x[length_size:]
            x_oe, lam = self.mixup_process(x_in, x_out)
            lam_return = lam
            x = torch.cat([x_in, x_oe], dim=0)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        last_feature = x.view(x.size(0), -1)
        x = self.fc(last_feature)
        return x, lam_return

    def manimix_process(self, x, y, alpha=0.3):
        index = np.random.permutation(x.size(0))
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_b = y[index]
        return mixed_x, y_b, lam

    def forward_manimix(self, x, y):
        layer_mix = np.random.randint(0, 3)
        if layer_mix == 0:
            x, y_b, lam = self.manimix_process(x, y)
        x = self.conv1(x)
        x = self.layer1(x)
        if layer_mix == 1:
            x, y_b, lam = self.manimix_process(x, y)
        x = self.layer2(x)
        if layer_mix == 2:
            x, y_b, lam = self.manimix_process(x, y)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        last_feature = x.view(x.size(0), -1)
        x = self.fc(last_feature)
        return x, y_b, lam


def resnet20(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet110(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model