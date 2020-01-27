import torch
import torch.nn as nn


class CrossEntropySmooth(torch.nn.Module):
    def __init__(self, classes_num, eps=0.1):
        super(CrossEntropySmooth, self).__init__()
        self.classes_num = classes_num
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, target):
        log = self.logsoftmax(inputs)
        place = torch.zeros_like(log).scatter_(
            1, torch.unsqueeze(target, dim=1), 1)
        qi = (1 - self.eps) * place + self.eps/self.classes_num
        loss = (-qi * log).mean(0).sum()
        return loss
