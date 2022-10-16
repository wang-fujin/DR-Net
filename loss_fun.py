from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from Nets import Discriminator

__all__ = ['AdversarialLoss', 'DiffLoss','InfoNCE']

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p=1.0):
        ctx.p = p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p
        return output, None


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class AdversarialLoss(LambdaSheduler):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(AdversarialLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        self.domain_classifier = Discriminator()

    def forward(self, source, target):
        lamb = self.lamb()
        self.step()
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss

    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class InfoNCE(nn.Module):

    def __init__(self,temperature=0.1,reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self,query,positive_key):
        return info_nce(query,positive_key,
                        temperature=self.temperature,
                        reduction=self.reduction)

def info_nce(query,positive_key,temperature,reduction):
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    query, positive_key = normalize(query, positive_key)
    logits = query @ transpose(positive_key)
    labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits/temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


if __name__ == '__main__':
    x1 = torch.randn(32, 256, requires_grad=True)
    x2 = torch.randn(32, 256, requires_grad=True)

    adv = AdversarialLoss()
    adv_loss = adv(x1,x2)

    print(adv_loss)




