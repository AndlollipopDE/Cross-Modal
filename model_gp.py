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


class PartBlock(nn.Module):
    def __init__(self):
        super(PartBlock, self).__init__()
        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)
        self.fc = nn.Linear(2048, 395)
        self.fc.apply(weights_init_classifier)
        return

    def forward(self, xt):
        xi = self.bn(xt)
        out = self.fc(xi)
        return xi, out


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
        self.pool = nn.AdaptiveAvgPool2d((6, 1))
        # Last Stride
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
        x_part = self.pool(x)
        x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x_part = torch.squeeze(x_part)
        #x = self.dropout(x)
        return x, x_part


class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, arch='resnet50'):
        super(embed_net, self).__init__()
        self.visible_net = visible_net_resnet(arch=arch)
        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)
        self.fc = nn.Linear(2048, 395)
        self.fc.apply(weights_init_classifier)
        self.parts_num = 6
        for i in range(self.parts_num):
            setattr(self, 'part'+str(i), PartBlock())

    def forward(self, x1, x2, modal=0, epoch=0):
        x = torch.cat((x1, x2), 0)
        yt, yt_part = self.visible_net(x)
        yi = self.bn(yt)
        out = self.fc(yi)
        yi_part = []
        out_part = []
        for i in range(self.parts_num):
            p = getattr(self, 'part'+str(i))
            part = torch.squeeze(yt_part[:, :, i])
            yi_temp, out_temp = p(part)
            yi_part.append(yi_temp)
            out_part.append(out_temp)
        if self.training:
            return out, yt, out_part, yt_part
        else:
            return yi, yi_part


# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)
