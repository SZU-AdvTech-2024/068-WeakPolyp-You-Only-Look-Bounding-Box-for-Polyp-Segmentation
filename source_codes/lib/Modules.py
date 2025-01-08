# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn

affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from lib.GatedConv import GatedConv2dWithActivation



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2  # 根据卷积核大小动态计算填充的边界大小。
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding) # 用于在输入特征图边界添加反射填充。
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class BasicConv2d(nn.Module):
    # Conv+norm+relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class BasicDeConv2d(nn.Module):
    # DeConv+norm+relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


"""
    Position Attention Module (PAM)
"""


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


"""
    ASPP
"""

class GPM(nn.Module):
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=128):
        # def __init__(self, dilation_series=[2, 5, 7], padding_series=[2, 5, 7], depth=128):
        super(GPM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(2048, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(2048, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth * 5, 256, kernel_size=3, padding=1),
            PAM(256)
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64, affine=affine_par),
            nn.PReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out


"""
    R-Net
"""


class ETM(nn.Module):
    # R-Net
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)
        # 1×1 卷积用于减少或调整通道数，同时保留空间信息，分支 0 提供了直接的通道映射，作为一种快速路径。
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        # 第一层是 1×1 卷积，用于调整通道数。
        # 第二、三层是方向敏感的卷积（1×3 和 3×1）。
        # 第四层是膨胀卷积，具有更大的感受野（膨胀系数为 3）。
        # 通过多个方向卷积，捕获特征的方向性信息。膨胀卷积在不增加计算量的情况下增大感受野。 增强模型对方向性和多尺度特征的捕获能力。
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5)
        ) # like branch1
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        ) # like branch1
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1)
        # 拼接的特征整合为统一的通道数，便于后续处理。残差连接提供快捷路径，防止梯度消失问题。
        # 通过特征融合和残差连接，既保留输入特征，又增强网络深度的学习能力。

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DWT(nn.Module):
    # 离散小波变换 Discrete Wavelet Transform
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
        # DWT 是一种特征变换操作，与模型的可学习参数无关，因此不需要梯度更新。

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2 #  对输入张量 x 的高度维度（第 3 维）每隔两行采样一行，并缩小值为原来的一半。
        x02 = x[:, :, 1::2, :] / 2 # 对输入张量 x 的高度维度每隔两行采样另一行，并缩小值为原来的一半。
        # 小波变换的第一步是对输入特征进行分块采样，分别保留偶数行和奇数行。将采样值缩小为原来的一半，符合小波变换的规范化处理。
        # 离散小波变换需要通过下采样操作提取局部区域的信息。
        x1 = x01[:, :, :, 0::2] # 从 x01 中提取宽度维度的偶数列。
        x2 = x02[:, :, :, 0::2] # 从 x02 中提取宽度维度的偶数列。
        x3 = x01[:, :, :, 1::2] # 从 x01 中提取宽度维度的奇数列。
        x4 = x02[:, :, :, 1::2] # 从 x02 中提取宽度维度的奇数列。
        # 分别对 x01 和 x02 的宽度维度进行采样，形成四个子特征块（x1、x2、x3、x4）。
        # 离散小波变换的目的是将原始图像分解为局部的子区域，每个子区域表示特定频域的信息。
        ll = x1 + x2 + x3 + x4      # 计算低频分量（全局信息）
        lh = -x1 + x2 - x3 + x4     # 计算水平高频分量（水平边缘信息）
        hl = -x1 - x2 + x3 + x4     # 计算垂直高频分量（垂直边缘信息）。
        hh = x1 - x2 - x3 + x4      # 计算对角高频分量（对角边缘信息）


        return ll, lh, hl, hh
        # 通过加减法操作对采样的四个子特征块进行组合，得到对应的频域信息。
        # 低频分量 ll 表示全局光照或纹理等平滑区域的信息。
        # 高频分量 lh、hl 和 hh 表示不同方向的边缘或细节信息。
        # 这种组合操作是小波变换的核心，用于提取特定频率的特征。

"""
    Joint Attention module (CA + SA)
    spatial attention channel attention
"""

class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, bias=True), # 将输入通道数降维到 channels // 4，减小计算量并提取局部特征
            nn.ReLU(inplace=True), # 引入非线性映射，增强模型表达能力。
            nn.Conv2d(channels // 4, 1, 3, padding=1, bias=True), # 将降维后的特征进一步映射为单通道特征图。
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.sa(x)
        # 通过两次卷积和激活操作，生成空间注意力权重图。
        # 使用注意力机制在空间维度上生成一个权重图，帮助网络关注重要区域，抑制无关区域。
        y = x * out
        return y

class CA(nn.Module):
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1) # 将特征图在空间维度上进行全局池化。
        # 平均池化聚合全局信息，适用于平滑特征。
        # 最大池化聚合显著信息，适用于突出激活区域。
        self.conv = nn.Conv1d(1, 1, kernel_size= 3, padding=(3 - 1) // 2, bias=False) # 一维卷积层，用于跨通道的信息交互和特征提取。
        self.sigmoid = nn.Sigmoid() # 激活函数，将输出值归一化到 [0, 1]，作为注意力权重。

    def forward(self, x):
        y = self.ap(x) # (N, C, 1, 1) # 将每个通道的空间特征聚合成一个单一的值，表示该通道的全局表示
        # 提取通道的全局信息，为后续通道间特征提取和加权提供依据。
        # 通过通道注意力机制，让网络能够动态关注不同通道的重要性，增强对目标区域的表达能力。
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y.squeeze(-1)：去除空间维度，形状从 (N, C, 1, 1) 变为 (N, C, 1)。
        # .transpose(-1, -2)：转置，形状从 (N, C, 1) 变为 (N, 1, C)，便于一维卷积处理。
        # self.conv(...)：使用一维卷积处理跨通道的信息。
        # .transpose(-1, -2).unsqueeze(-1)：恢复原始形状，最终输出形状为 (N, C, 1, 1)。
        # 一维卷积通过滑动窗口操作捕获通道间的关系，生成每个通道的注意力权重。
        # 转置操作确保卷积在通道维度上进行操作，而不是批量或特征维度。
        # 通过一维卷积提取通道间的上下文关系，进一步优化注意力分布。
        y = self.sigmoid(y)
        return x * y.expand_as(x) # y.expand_as(x)：将通道注意力权重扩展到与输入 x 相同的形状。


class AM(nn.Module):
    # CA+SA channel attention + spatial attention
    def __init__(self, channels, lf):
        super(AM, self).__init__()
        self.CA = CA(lf=lf)
        self.SA = SA(channels)

    def forward(self, x):
        x = self.CA(x)
        x = self.SA(x)
        return x


"""
    Low-Frequency Attention Module (LFA)
"""


class RB(nn.Module):
    def __init__(self, channels, lf):
        super(RB, self).__init__()
        self.RB = BasicConv2d(channels, channels, 3, padding=1, bn=nn.InstanceNorm2d if lf else nn.BatchNorm2d)
        # Conv+norm+relu

    def forward(self, x):
        y = self.RB(x)
        return y + x


class ARB(nn.Module):
    # 若lf=0，则为HFA（CA+SA+Res），若lf=1，则为LFA（相比多了个归一化）
    def __init__(self, channels, lf):
        super(ARB, self).__init__()
        self.lf = lf
        self.AM = AM(channels, lf) # CA+SA
        self.RB = RB(channels, lf) # 3x3卷积+norm+激活+残差连接

        self.mean_conv1 = ConvLayer(1, 16, 1, 1)
        self.mean_conv2 = ConvLayer(16, 16, 3, 1)
        self.mean_conv3 = ConvLayer(16, 1, 1, 1)

        self.std_conv1 = ConvLayer(1, 16, 1, 1)
        self.std_conv2 = ConvLayer(16, 16, 3, 1)
        self.std_conv3 = ConvLayer(16, 1, 1, 1)

    def PONO(self, x, epsilon=1e-5):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
        output = (x - mean) / std
        return output, mean, std

    def forward(self, x):
        if self.lf:
            x, mean, std = self.PONO(x)
            mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
            std = self.std_conv3(self.std_conv2(self.std_conv1(std)))
        y = self.RB(x)
        y = self.AM(y)
        if self.lf:
            return y * std + mean
        return y


"""
    Guidance-based Upsampling
"""

class BoxFilter(nn.Module):
    # 实现快速滑动窗口滤波器
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r # 滑动窗口半径，决定滤波器的感受野大小
        # 滤波器的大小为 (2r + 1) × (2r + 1)

    def diff_x(self, input, r):
        assert input.dim() == 4

        left = input[:, :, r:2 * r + 1]
        middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
        right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=2)

        return output

    def diff_y(self, input, r):
        assert input.dim() == 4

        left = input[:, :, :, r:2 * r + 1]
        middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
        right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]

        output = torch.cat([left, middle, right], dim=3)

        return output

    def forward(self, x):
        assert x.dim() == 4
        return self.diff_y(self.diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GF(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GF, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A * hr_x + mean_b).float()


"""
    Guidance-based Feature Aggregation Module (GFA)
"""


class AGF(nn.Module):
    def __init__(self, channels, lf):
        super(AGF, self).__init__()
        self.ARB = ARB(channels, lf) # 一个特征增强模块，增强输入的低层特征。
        self.GF = GF(r=2, eps=1e-2) # 一个特征融合模块，用于融合高层和低层特征。

    def forward(self, high_level, low_level):
        N, C, H, W = high_level.size()
        high_level_small = F.interpolate(high_level, size=(int(H / 2), int(W / 2)), mode='bilinear', align_corners=True) # 下采样2x
        y = self.ARB(low_level)
        y = self.GF(high_level_small, low_level, high_level, y)
        # 高层特征和低层特征在表达能力上有所不同，直接融合可能会导致信息冲突。
        # 通过先增强低层特征，再融合高层特征，能有效提高特征表示的质量。
        return y


class AGFG(nn.Module):
    # 用于特定的特征融合和增强任务。
    def __init__(self, channels, lf):
        super(AGFG, self).__init__()
        self.GF1 = AGF(channels, lf)
        self.GF2 = AGF(channels, lf)
        self.GF3 = AGF(channels, lf)

    def forward(self, f1, f2, f3, f4):
        y = self.GF1(f2, f1)
        y = self.GF2(f3, y)
        y = self.GF3(f4, y)
        return y



"""
    Deep Wavelet-like Decomposition (DWD): LWD + HFA/LFA + GFA
"""


class GCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        GCM3 模块通过多尺度特征提取、小波分解和注意力机制，提取图像的多频域信息并增强其表征能力。
        这样的设计使得模型在处理分割任务时能够同时关注全局结构和局部细节，特别适合对边界信息要求高的任务。
        '''
        super(GCM3, self).__init__()
        self.T1 = ETM(in_channels, out_channels)
        self.T2 = ETM(in_channels * 2, out_channels)
        self.T3 = ETM(in_channels * 4, out_channels)
        self.T4 = ETM(in_channels * 8, out_channels)

        # wavelet attention module
        self.DWT = DWT()
        self.AGFG_LL = AGFG(out_channels, True) # 特征融合，低层细节高层语义
        self.AGFG_LH = AGFG(out_channels, False)
        self.AGFG_HL = AGFG(out_channels, False)
        self.AGFG_HH = AGFG(out_channels, False)

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)

        wf1 = self.DWT(f1)
        wf2 = self.DWT(f2)
        wf3 = self.DWT(f3)
        wf4 = self.DWT(f4)
        # 返回低频 LL，水平高频 LH，垂直高频 HL，对角高频 HH

        LL = self.AGFG_LL(wf4[0], wf3[0], wf2[0], wf1[0])
        LH = self.AGFG_LH(wf4[1], wf3[1], wf2[1], wf1[1])
        HL = self.AGFG_HL(wf4[2], wf3[2], wf2[2], wf1[2])
        HH = self.AGFG_HH(wf4[3], wf3[3], wf2[3], wf1[3])
        return LL, LH, HL, HH, f1, f2, f3, f4



"""
    TFD
"""


class TFD(nn.Module):
    def __init__(self, in_channels):
        super(TFD, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.gatedconv = GatedConv2dWithActivation(in_channels * 2, in_channels, kernel_size=3, stride=1,
                                                   padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
                                                   activation=torch.nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feature_map, perior_repeat):
        assert (feature_map.shape == perior_repeat.shape), "feature_map and prior_repeat have different shape"
        uj = perior_repeat
        uj_conv = self.conv(uj)
        uj_1 = uj_conv + uj
        uj_i_feature = torch.cat([uj_1, feature_map], 1)
        uj_2 = uj_1 + self.gatedconv(uj_i_feature) - 3 * uj_conv
        return uj_2


"""
    Ordinary Differential Equation (ODE)
"""


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        alpha = self.getalpha(torch.cat([f1, f2], dim=1))
        out = feature_map + f1 * alpha + f2 * (1 - alpha)
        return out

"""
    segmentation-oriented edge-assited decoder (SED)
"""

class REU6(nn.Module):
    # 旨在结合输入特征和边缘信息，进行多阶段处理，生成分割结果和边缘图。
    def __init__(self, in_channels, mid_channels):
        super(REU6, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.TFD = TFD(in_channels)
        self.out_B = nn.Sequential(
            BasicDeConv2d(in_channels, mid_channels // 2, kernel_size=3, stride=2, padding=1, out_padding=1),
            BasicConv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.ode = ODE(in_channels)

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=x.size()[2:], mode='bilinear', align_corners=True)  # 2,1,12,12->2,1,48,48
        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size()[1], -1, -1)], dim=1))

        ode_out = self.ode(yt) # 模块处理融合特征
        bound = self.out_B(ode_out) # 生成边界检测结果
        bound = self.edge_enhance(bound) # 增强边界

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y = r_prior_cam.expand(-1, x.size()[1], -1, -1).mul(x)
        # 计算负先验掩码，用于屏蔽非目标区域的特征
        # 对目标区域以外的特征赋予较低权重
        # 通过负先验信息抑制干扰特征

        cat2 = torch.cat([y, ode_out], dim=1)  # 2,128,48,48
        # 将负先验修正后的特征与 ODE 输出拼接
        y = self.out_y(cat2) # 使用 out_y 模块生成最终分割图
        y = y + prior_cam # 将先验掩码加到输出结果中，进一步强化目标区域
        return y, bound

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class REM11(nn.Module):
    # 通过逐层递归的方式处理多尺度特征，同时生成目标分割图和边界检测结果。
    def __init__(self, in_channels, mid_channels):
        super(REM11, self).__init__()
        self.REU_f1 = REU6(in_channels, mid_channels)
        self.REU_f2 = REU6(in_channels, mid_channels)
        self.REU_f3 = REU6(in_channels, mid_channels)
        self.REU_f4 = REU6(in_channels, mid_channels)
        # REU6 是一个增强模块，结合了特征融合、先验信息和边界检测。
        # 每个模块专注于特定的特征层级
        # 通过分层处理，捕捉不同尺度的特征，有助于提升分割精度。

    def forward(self, x, prior_0, pic):
        f1, f2, f3, f4 = x # f1 到 f4 表示从浅层到深层的特征，深层特征通常包含更抽象的语义信息
        # 分层处理特征有助于模型分别关注低级（细节）和高级（语义）信息

        f4_out, bound_f4 = self.REU_f4(f4, prior_0)  # b,1,12,12 b,1,48,48
        f4 = F.interpolate(f4_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f4 = F.interpolate(bound_f4, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)  # b,1,24,24 b,1,96,96
        f3 = F.interpolate(f3_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f3 = F.interpolate(bound_f3, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)  # b,1,48,48 b,1,192,192
        f2 = F.interpolate(f2_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f2 = F.interpolate(bound_f2, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)  # b,1,96,96 b,1,384,384
        f1 = F.interpolate(f1_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f1 = F.interpolate(bound_f1, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        return f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


if __name__ == '__main__':
    f1 = torch.randn(2, 1, 12, 12).cuda()
    ll = torch.randn(2, 64, 96, 96).cuda()
    lh = torch.randn(2, 64, 48, 48).cuda()
    hl = torch.randn(2, 64, 24, 24).cuda()
    hh = torch.randn(2, 64, 12, 12).cuda()
    pict = torch.randn(2, 3, 384, 384).cuda()
    x = [ll, lh, hl, hh]
    rem = REM11(64, 64).cuda()
    f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = rem(x, f1, pict)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)
