import torch
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, batchsize, num_instances, margin=0.3, mutual_flag=False, cross_modal=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag
        self.cross_modal = cross_modal
        self.batchsize = batchsize
        self.num_instances = num_instances
        self.num_ids = self.batchsize / self.num_instances

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        if not self.cross_modal:
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        else:
            for i in range(n):
                if i >= (n / 2):
                    dist_ap.append(
                        dist[i][mask[i]][0:self.num_instances].max().unsqueeze(0))
                    dist_an.append(
                        dist[i][mask[i] == 0][0:self.batchsize-self.num_instances].min().unsqueeze(0))
                else:
                    dist_ap.append(
                        dist[i][mask[i]][self.num_instances:].max().unsqueeze(0))
                    dist_an.append(
                        dist[i][mask[i] == 0][self.batchsize-self.num_instances:].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
