import torch
import torch.nn as nn


class CovLoss(nn.Module):
    def __init__(self, batchsize, num_instance, margin=0.00):
        super(CovLoss, self).__init__()
        self.margin = margin
        self.batchsize = batchsize
        self.num_instance = num_instance
        self.relu = nn.ReLU()
        return

    def forward(self, feat):
        feat = self.relu(feat)
        loss = self.compute_indep(feat)
        return loss

    def compute_indep(self, feat):
        feat_dim = feat.size(1)
        mask = torch.eye(feat_dim).cuda()
        feat_mean = torch.mean(feat, dim=0, keepdim=True)
        feat_centerless = feat - feat_mean
        feat_covar = torch.matmul(feat_centerless.t(), feat_centerless)
        feat_var = feat_covar.diag().unsqueeze(1)
        feat_cvar = torch.matmul(
            feat_var, feat_var.t()).clamp(min=1e-12).sqrt()
        feat_dep = torch.div(feat_covar, feat_cvar).abs()
        masked_dep = torch.masked_select(feat_dep, mask == 0)
        loss = torch.max(torch.zeros_like(masked_dep),
                         masked_dep-self.margin).mean()
        return loss

    def compute_dep(self, feat1, feat2):
        feat1_dim = feat1.size(1)
        feat1_mean = torch.mean(feat1, dim=0, keepdim=True)
        feat2_mean = torch.mean(feat2, dim=0, keepdim=True)
        feat1_cl = feat1 - feat1_mean
        feat2_cl = feat2 - feat2_mean
        feat_covar = torch.matmul(feat1_cl.t(), feat2_cl).diag()
        feat1_var = torch.matmul(feat1_cl.t(), feat1_cl).diag()
        feat2_var = torch.matmul(feat2_cl.t(), feat2_cl).diag()
        feat_var = torch.mul(feat1_var, feat2_var).clamp(min=1e-12).sqrt()
        feat_dep = torch.div(feat_covar, feat_var)
        mask = torch.ones_like(feat_dep)
        loss = (mask - feat_dep).mean()
        return loss
