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


# #####################################################################

# Non-Local attention


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2, se_ratio=16):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.seinter_channels = self.in_channels // se_ratio
        self.SE = nn.Sequential(
            nn.Conv2d(self.in_channels, self.seinter_channels, 1),
            nn.ReLU(),
            nn.Conv2d(self.seinter_channels, self.in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = torch.nn.functional.softmax(f, dim=-1)
        # f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        #W_y_cw = self.SE(W_y)
        #W_y = torch.mul(W_y,W_y_cw)
        z = W_y + x

        return z


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
        layers = [3, 4, 6, 3]
        num_layer = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList([Non_local(256)
                                   for i in range(num_layer[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1)
                                for i in range(num_layer[0])])
        self.NL_2 = nn.ModuleList([Non_local(512)
                                   for i in range(num_layer[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1)
                                for i in range(num_layer[1])])
        self.NL_3 = nn.ModuleList([Non_local(1024)
                                   for i in range(num_layer[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1)
                                for i in range(num_layer[2])])
        self.NL_4 = nn.ModuleList([Non_local(2048)
                                   for i in range(num_layer[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1)
                                for i in range(num_layer[3])])
        # self.avgpool = nn.AdaptiveAvgPool2d((6,1))

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        counter1 = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.visible.layer1)):
            x = self.visible.layer1[i](x)
            if i == self.NL_1_idx[counter1]:
                x = self.NL_1[counter1](x)
                counter1 += 1
        counter2 = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.visible.layer2)):
            x = self.visible.layer2[i](x)
            if i == self.NL_2_idx[counter2]:
                x = self.NL_2[counter2](x)
                counter2 += 1
        counter3 = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.visible.layer3)):
            x = self.visible.layer3[i](x)
            if i == self.NL_3_idx[counter3]:
                x = self.NL_3[counter3](x)
                counter3 += 1
        counter4 = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.visible.layer4)):
            x = self.visible.layer4[i](x)
            if i == self.NL_4_idx[counter4]:
                x = self.NL_4[counter4](x)
                counter4 += 1
        x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        # x = self.dropout(x)
        return x


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

    def forward(self, x1):
        if self.weight:
            yt, w = self.backbone(x1)
        else:
            yt = self.backbone(x1)
        yi = self.bn(yt)
        out = self.fc(yi)
        if self.weight:
            return out, yt, yi, w
        else:
            return out, yt, yi

# debug model structure

# net = embed_net(512, 319)
# net.train()
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# x, y  = net(input, input)
