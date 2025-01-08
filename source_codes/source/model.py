import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import Res2Net50
from pvtv2 import pvt_v2_b2
from utils import weight_init
from lib.Modules import GCM3,BasicConv2d
from lib.Network import Network

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.linear1 = nn.Sequential(nn.Conv2d(channels[0], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64))

    def forward(self, x1, x2, x3, x4):
        X1, x2, x3, x4   = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4           = F.interpolate(x4, size=x2.size()[2:], mode='bilinear')
        x3           = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        out          = x2*x3*x4
        return out

    def initialize(self):
        weight_init(self)

class WeakPolyp(nn.Module):
    def __init__(self, cfg):
        super(WeakPolyp, self).__init__()
        if cfg.backbone=='res2net50':
            self.backbone = Network(96)
        if cfg.backbone=='pvt_v2_b2':
            self.backbone = pvt_v2_b2()


    def forward(self, x):
        pred = self.backbone(x)[4]
        return pred