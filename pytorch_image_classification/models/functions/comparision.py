#  Copyright (c) 2018, TU Darmstadt.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

###
class pospowbias(nn.Module):
    def __init__(self, in_channels):
        super(pospowbias, self).__init__()
        self.Lambda = nn.Parameter(torch.rand(in_channels))
        self.Alpha = nn.Parameter(torch.rand(in_channels))
        self.channels = in_channels

    def forward(self, x):
        o = x.pow(self.Lambda.exp().reshape(1, self.channels, 1, 1))
        o = o + self.Alpha.exp().reshape(1, self.channels, 1, 1)
        return o

class DPP(nn.Module):
    def __init__(self, in_channels):
        super(DPP, self).__init__()
        self.pospowbias = pospowbias(in_channels)
        self.avgpool = nn.AvgPool2d(2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        print("DPP")
    def forward(self, I):
        It = self.upsample(self.avgpool(I))
        x = ((I-It)**2)+1e-3
        xn = self.upsample(self.avgpool(x))
        w = self.pospowbias(x/xn)
        kp = self.avgpool(w)
        Iw = self.avgpool(I*w)
        return Iw/kp
class DPP_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DPP_AlterChan, self).__init__()
        self.dpp = DPP(in_channels=in_channels)
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.dpp(x)
        x = self.alter_chan(x)
        return x




BOTTLENECK_WIDTH = 128
COEFF = 12.0


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    weight = logit.exp()
    return F.avg_pool2d(x*weight, kernel, stride, padding)/F.avg_pool2d(weight, kernel, stride, padding)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SoftGate(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


class BottleneckLIP(nn.Module):
    def __init__(self, channels):
        super(BottleneckLIP, self).__init__()

        rp = BOTTLENECK_WIDTH

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv1x1(channels, rp)),
                ('bn1', nn.InstanceNorm2d(rp, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(rp, rp)),
                ('bn2', nn.InstanceNorm2d(rp, affine=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', conv1x1(rp, channels)),
                ('bn3', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )

    def init_layer(self):
        self.logit[6].weight.data.fill_(0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x), kernel=2, stride=2, padding=0)
        return frac
class BottleneckLIP_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckLIP_AlterChan, self).__init__()
        self.bottleneckLIP = BottleneckLIP(channels=in_channels)
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.bottleneckLIP(x)
        x = self.alter_chan(x)
        return x

class SimplifiedLIP(nn.Module):
    def __init__(self, channels):
        super(SimplifiedLIP, self).__init__()

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv1', conv3x3(channels, channels)),
                ('bn1', nn.InstanceNorm2d(channels, affine=True)),
                ('gate', SoftGate()),
            ))
        )
        print("simplified LIP")

    def init_layer(self):
        self.logit[0].weight.data.fill_(0)

    def forward(self, x):
        frac = lip2d(x, self.logit(x), kernel=3, stride=2, padding=1)
        return frac


from torch.nn import LPPool2d

class MaxPooling_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPooling_AlterChan, self).__init__()
        self.max = nn.MaxPool2d(2)
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.max(x)
        x = self.alter_chan(x)
        return x

class AvgPooling_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AvgPooling_AlterChan, self).__init__()
        self.avg = nn.AvgPool2d(2)
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        x = self.avg(x)
        x = self.alter_chan(x)
        return x
        
class AvgPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AvgPooling, self).__init__()
        self.avg = nn.AvgPool2d(2)
    
    def forward(self, x):
        x = self.avg(x)
        return x

class MixPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.rand(1))
        self.max = nn.MaxPool2d(2)
        self.avg = nn.AvgPool2d(2)
        print("mix pooling")

    def forward(self, x):
        return self.alpha * self.max(x) + (1 - self.alpha) * self.avg(x)
class MixPooling_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixPooling_AlterChan, self).__init__()
        self.mixpool = MixPooling()
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.mixpool(x)
        x = self.alter_chan(x)
        return x

class GatedPool(nn.Module):
    def __init__(self, in_channel, kernel_size=2, stride=2):
        super(GatedPool, self).__init__()
        self.w = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride)
        self.sigmoid = nn.Sigmoid()
        self.max = nn.MaxPool2d(2)
        self.avg = nn.AvgPool2d(2)
        print("gated pool")

    def forward(self, x):
        alpha = self.sigmoid(self.w(x))
        return alpha * self.max(x) + (1 - alpha) * self.avg(x)
class GatedPool_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedPool_AlterChan, self).__init__()
        self.gatedpool = GatedPool(in_channels)
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.gatedpool(x)
        x = self.alter_chan(x)
        return x

import pywt

class DWT(nn.Module):
    def __init__(self, wavelet):
        # decice is not needed anymore
        super(DWT, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet)
        self.lp_band = torch.FloatTensor(self.wavelet.rec_lo)
        self.hp_band = torch.FloatTensor(self.wavelet.rec_hi)
        self.band_length = len(self.hp_band)
        self.half_band_length = self.band_length//2
        #print("dwt")

    def gene_matrix(self, device):
        edge_len = self.h
        half_edge_len = edge_len//2
        matrix_h = torch.zeros((half_edge_len, edge_len + self.band_length - 2), requires_grad=False)
        matrix_g = torch.zeros((half_edge_len, edge_len + self.band_length - 2), requires_grad=False)
        index = 0
        for i in range(half_edge_len):
            matrix_h[i, index:index+self.band_length] = self.lp_band
            index += 2
        index = 0
        for i in range(half_edge_len):
            matrix_g[i, index:index+self.band_length] = self.hp_band
            index += 2
        end = None if self.half_band_length == 1 else -self.half_band_length+1
        matrix_h = matrix_h[:, self.half_band_length-1: end].to(device)
        matrix_g = matrix_g[:, self.half_band_length-1: end].to(device)
        return matrix_h, matrix_g

    def forward(self, x):
        self.h, self.w = x.shape[2], x.shape[3]
        M_L, M_H = self.gene_matrix(x.device)
        L = torch.matmul(M_L, x)
        H = torch.matmul(M_H, x)
        LL = torch.matmul(L, M_L.transpose(-1, -2))
        LH = torch.matmul(L, M_H.transpose(-1, -2))
        HL = torch.matmul(H, M_L.transpose(-1, -2))
        HH = torch.matmul(H, M_H.transpose(-1, -2))
        return LL, LH, HL, HH

class DWTPooling(nn.Module):
    def __init__(self):
        super(DWTPooling, self).__init__()

    def forward(self, x):
        xll, xlh, xhl, xhh = DWT("haar")(x)
        return xll
class DWTPooling_AlterChan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWTPooling_AlterChan, self).__init__()
        self.dwtpool = DWTPooling()
        self.alter_chan = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.dwtpool(x)
        x = self.alter_chan(x)
        return x

# Soft Pool: https://git.io/JL5zL
# or https://github.com/alexandrosstergiou/SoftPool
# blog: https://blog.csdn.net/WZZ18191171661/article/details/113048529
from torch.autograd import Function

class CUDA_SOFTPOOL2d(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.size()
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)
        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1
        output = input.new_zeros((B, C, oH, oW))
        softpool_cuda.forward_2d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel,ctx.stride] + [grad_input]
        softpool_cuda.backward_2d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None

def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))

class SoftPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)
