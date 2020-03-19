import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import random


# L2 normalized
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)


# Define the ResNet18-based Model
class backbone(nn.Module):
    def __init__(self, arch='resnet18'):
        super(backbone, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.layer4[0].downsample[0].stride = (1, 1)
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
        self.SE1 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1024, 1),
            nn.Sigmoid()
        )
        self.SE2 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x3_pool = self.visible.avgpool(x)
        x3_cmap = self.SE1(x3_pool)
        x3_cwei = torch.mul(x3_cmap, x)
        x3 = self.visible.avgpool(x3_cwei)
        x3 = x3.view(x.size(0), x.size(1))

        x = self.visible.layer4(x)
        x_pool = self.visible.avgpool(x)
        x_cmap = self.SE2(x_pool)
        x_cwei = torch.mul(x_cmap, x)
        x = self.visible.avgpool(x_cwei)
        x = x.view(x.size(0), x.size(1))

        return x, x3


class backbone_weight(nn.Module):
    def __init__(self, arch='resnet18'):
        super(backbone_weight, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.layer4[0].downsample[0].stride = (1, 1)
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
        # self.avgpool = nn.AdaptiveAvgPool2d((6,1))

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x = self.visible.layer4(x)
        weight = x[:, -1, :, :]
        weight = torch.unsqueeze(weight, 1)
        x = torch.mul(x[:, :2047, :, :], weight)
        x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x, weight


class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, arch='resnet50', weight_flag=False):
        super(embed_net, self).__init__()
        self.weight = weight_flag
        if self.weight:
            self.backbone = backbone_weight(arch=arch)
        else:
            self.backbone = backbone(arch=arch)

        self.dim = low_dim
        self.bn = nn.BatchNorm1d(self.dim)
        self.bn.apply(weights_init_kaiming)
        self.fc = nn.Linear(self.dim, class_num)
        self.fc.apply(weights_init_classifier)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn3.apply(weights_init_kaiming)
        self.fc3 = nn.Linear(1024, class_num)
        self.fc3.apply(weights_init_classifier)

    def forward(self, x1):
        if self.weight:
            yt, w = self.backbone(x1)
        else:
            yt, yt3 = self.backbone(x1)
        yi = self.bn(yt)
        out = self.fc(yi)
        yi3 = self.bn3(yt3)
        out3 = self.fc3(yi3)
        if self.weight:
            return out, yt, yi, w
        else:
            return out, yt, yi, out3, yt3, yi3

# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)
