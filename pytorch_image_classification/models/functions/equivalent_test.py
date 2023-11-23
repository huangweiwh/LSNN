import torch
import torch.nn as nn
from lifting_scheme import LiftingScheme
import torch.nn.functional as F



if __name__ == '__main__':
    h = torch.randn((8, 4, 3, 3)).to('cuda')
    x = torch.ones((2, 4, 6, 6)).to('cuda')

    # out = F.conv2d(x, h, stride=2, padding=1)
    out = F.conv2d(x, h, stride=1, padding=1)
    print("convolution result:\n", out.shape)

    # perform the equivalent lifting scheme
    out_channels, in_channels, kernal_size, _ = h.shape
    ls = LiftingScheme(out_channels=out_channels, in_channels=in_channels,kernal_size = kernal_size,weight=h,mode='vanilla')
    # ls = LiftingScheme(h,mode='strided')
    result = ls(x)
    # print(result)
    print("lifting scheme result:\n", result.shape)

    assert abs(out - result).sum() < pow(10, -3)
    print("Correct!")
