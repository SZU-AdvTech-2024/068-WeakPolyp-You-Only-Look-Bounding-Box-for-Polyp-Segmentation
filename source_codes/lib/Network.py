import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.Modules import GCM3, GPM, REM11, BasicConv2d
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F

'''
backbone: resnet50
'''


class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=96):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=False, in_chans=3, features_only=True)
        # 创建一个基于 ResNet-50 的特征提取器，输入通道数为 3（RGB 图像）。
        # 参数 features_only=True 表示只返回中间特征，不包括分类头。一共有5个尺度的特征，大小分别是其
        self.GCM3 = GCM3(256, channels) # backbone result convert to output feature that will be input into decoder
        self.GPM = GPM() # from f4 get coarse segmentation result
        self.REM11 = REM11(channels, channels) # segmentation-oriented edge-assisted decoder

        self.LL_down = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        ) # 下采样4x
        self.dePixelShuffle = torch.nn.PixelShuffle(2) # 上采样2x
        self.one_conv_f4_ll = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        # x0 64 192 192
        # x1 256 96 96
        # x2 512 48 48
        # x3 1024 24 24
        # x4 2048 12 12

        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)
        # HH HL LH LL 96 48 48
        # f1 96 96 96
        # f2 96 48 48
        # f3 96 24 24
        # f4 96 12 12
        HH_up = self.dePixelShuffle(HH)  
        # 24 96 96
        f1_HH = torch.cat([HH_up, f1], dim=1)
        # 120 96 96
        f1_HH = self.one_conv_f1_hh(f1_HH)
        # 96 96 96

        LL_down = self.LL_down(LL)
        # 96 12 12
        f4_LL = torch.cat([LL_down, f4], dim=1)
        # 192 12 12
        f4_LL = self.one_conv_f4_ll(f4_LL)
        # 96 12 12

        prior_cam = self.GPM(x4) # e-ASPP
        # 1 12 12
        pred_0 = F.interpolate(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        # 1 384 384

        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1_HH, f2, f3, f4_LL], prior_cam, image)
        # all 1 384 384
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1
        # f1是最终结果，其损失的权重是最大的，并且test的时候也是f1


if __name__ == '__main__':
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network(32).cuda()
    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print('pred_0 shape:',pred_0.shape)
    print('f4 shape:',f4.shape)
    print('f3 shape:',f3.shape)
    print('f2 shape:',f2.shape)
    print('f1 shape:',f1.shape)
    print('bound_f4 shape:',bound_f4.shape)
    print('bound_f3 shape:',bound_f3.shape)
    print('bound_f2 shape:',bound_f2.shape)
    print('bound_f1 shape:',bound_f1.shape)

