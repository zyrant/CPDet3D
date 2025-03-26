# https://github.com/tfzhou/ProtoSeg
import torch
import torch.nn as nn
import torch.nn.functional as F


def momentum_update(old_value, new_value, momentum, iter, warm_up, debug=False):
    # iter = iter - warm_up
    # momentum = min(1 - 1 / ((iter/10) + 1), momentum)
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result-row= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, proj_dim))

    def forward(self, x):
        return l2_normalize(self.proj(x))