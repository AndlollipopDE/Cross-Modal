import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import random


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

# #####################################################################


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
class visible_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(visible_net_resnet, self).__init__()
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
        #self.avgpool = nn.AdaptiveAvgPool2d((6,1))

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
        #x = self.dropout(x)
        return x, weight


class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, arch='resnet50'):
        super(embed_net, self).__init__()
        if arch == 'resnet18':
            self.visible_net = visible_net_resnet(arch=arch)
            pool_dim = 512
        elif arch == 'resnet50':
            self.visible_net = visible_net_resnet(arch=arch)
            pool_dim = 2048

        self.bn = nn.BatchNorm1d(2047)
        self.bn.apply(weights_init_kaiming)
        self.fc = nn.Linear(2047, 395)
        self.fc.apply(weights_init_classifier)

    def forward(self, x1, x2, modal=0, epoch=0):
        if modal == 0:
            x = torch.cat((x1, x2), 0)
            yt, weight = self.visible_net(x)
            yi = self.bn(yt)
            out = self.fc(yi)

        elif modal == 1:
            yt, weight = self.visible_net(x1)
            yi = self.bn(yt)
        elif modal == 2:
            yt, weight = self.visible_net(x2)
            yi = self.bn(yt)

        if self.training:
            return out, yt, yi, weight
        else:
            return yi


# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)
