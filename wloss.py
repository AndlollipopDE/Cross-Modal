import torch
from torch import nn


def normalize(inputs):
    temp = 1.*inputs / \
        (torch.norm(inputs, 2, 1, keepdim=True).expand_as(inputs) + 1e-10)
    return temp


class WLoss(nn.Module):
    def __init__(self, batchsize, num_instances, margin=1, normalize=True):
        super(WLoss, self).__init__()
        self.margin = margin
        self.batchsize = batchsize
        self.num_instances = num_instances
        return

    def forward(self, inputs, targets):
        inputs = torch.squeeze(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = normalize(inputs)
        n = inputs.size(0)
        #dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        #dist = dist + dist.t()
        #dist.addmm_(1, -2, inputs, inputs.t())
        #dist = dist.clamp(min=1e-12).sqrt()
        dist = torch.mm(inputs, inputs.t())
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        loss = 0.0
        for i in range(n):
            if i >= (n / 2):
                dist_ap.append(
                    self.margin - dist[i][mask[i]][0:self.num_instances].min().unsqueeze(0))
                dist_an.append(
                    self.margin - dist[i][mask[i]][self.num_instances:].min().unsqueeze(0))
            else:
                dist_ap.append(
                    self.margin - dist[i][mask[i]][self.num_instances:].min().unsqueeze(0))
                dist_an.append(
                    self.margin - dist[i][mask[i]][0:self.num_instances].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.zeros_like(dist_an)
        loss1 = torch.max(y, dist_ap)
        loss2 = torch.max(y, dist_an)
        loss = torch.sum(loss1) + torch.sum(loss2)
        return loss
