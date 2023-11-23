'''
Proof the equivalence between the lifting     
scheme and the 2D convolution operation.
'''

# Author: Huang Wei and SHI Zishan
# Date: 2023/9/22

import torch
import torch.nn as nn


class LiftingScheme(nn.Module):
    def __init__(self, out_channels, in_channels, kernal_size, padding=1, channel_split=2, weight ='None', mode="vanilla"):
        """
        inputs:
        - in_channels
        - out_channels
        - kernal_size
        - padding
        - channel_split: for accumulate calculation , should be divided into out_channels
        - weight: tensor, filter kernel
        - mode: str, vanilla/strided
        """
        super(LiftingScheme, self).__init__()
        if weight == 'None':
            weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernal_size, kernal_size),requires_grad=True)
            nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
        self.weight = weight

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.padding = padding
        self.channel_split = channel_split

        self.mode = mode
        if self.mode not in ["vanilla", "strided"]:
            raise ValueError('mode must be "vanilla" or "strided"')

    def extract_param_p(self):
        """
        This function extract the predict filter from convolutional weights.

        need:
        - weight: tensor, weight tensor of the parameters of the filter kernels

        output:
        - p: tensor, predict filter P=[p0, p1, p2, p3]
        p0 forms p1(z) in the paper
        p1 forms p2(z) in the paper
        p2 and p3 together form the p3(z) in the paper
        """
        weight = self.weight

        p = torch.zeros([self.out_channels, self.in_channels, 4],device=torch.device("cuda"))

        p[:, :, 0] = - weight[:, :, 2, 0] / weight[:, :, 2, 1]
        p[:, :, 1] = - weight[:, :, 0, 2] / weight[:, :, 1, 2]
        p[:, :, 2] = (1 - weight[:, :, 0, 0] + weight[:, :, 2, 0] * weight[:, :, 0, 1] /
                   weight[:, :, 2, 1] + weight[:, :, 0, 2] * weight[:, :, 1, 0] / weight[:, :, 1, 2]) / weight[:, :, 1, 1]
        p[:, :, 3] = - weight[:, :, 2, 2] / weight[:, :, 1, 1]
        return p

    def extract_param_u(self):
        """
        This function extract the update filter from convolutional weights.

        need:
        - weight: tensor, weight tensor of the parameters of the filter kernels

        output:
        - u: tensor, update filter U=[u0, u1, u2, u3, u4]
        u0 and u1 form u1(z) in the paper
        u2 and u3 form u2(z) in the paper
        u4 forms u3(z) in the paper
        """
        weight = self.weight

        u = torch.zeros([self.out_channels, self.in_channels, 5],device=torch.device("cuda"))

        u[:, :, 0] = weight[:, :, 0, 1]
        u[:, :, 1] = weight[:, :, 2, 1]
        u[:, :, 2] = weight[:, :, 1, 0]
        u[:, :, 3] = weight[:, :, 1, 2]
        u[:, :, 4] = weight[:, :, 1, 1]

        return u

    def lifting_scheme_2d(self, x, p, u):
        """
        This function performs the lifting scheme that 
        equivalent to the 2D convolution.

        input:
        - x: tensor, input signal
        - p: tensor, predict filter
        - u: tensor, update filter

        output:
        - result: tensor, result signal from the lifting scheme
        """
        aux_index = 2 if self.mode == "vanilla" else 1
        # -------------- split -----------------------
        if self.mode == "vanilla":
            x0, x1 = torch.zeros(x.shape,device=torch.device("cuda")), torch.zeros(x.shape,device=torch.device("cuda"))
            x2, x3 = torch.zeros(x.shape,device=torch.device("cuda")), torch.zeros(x.shape,device=torch.device("cuda"))
            x0[:, :, :, :, :], x1[:, :, :, :, :-1] = x[:, :, :, :, :], x[:, :, :, :, 1:]
            x2[:, :, :, :-1, :], x3[:, :, :, :-1, :-1] = x[:, :, :, 1:, :], x[:, :, :, 1:, 1:]

        elif self.mode == "strided":
            x0, x1 = x[:, :, :, ::2, ::2], x[:, :, :, ::2, 1::2]
            x2, x3 = x[:, :, :, 1::2, ::2], x[:, :, :, 1::2, 1::2]

        # ---------------- predict --------------------
        x0_2 = torch.zeros(x0.shape,device=torch.device("cuda"))
        x0_2[:, :, :, :-aux_index, :-aux_index] = x0[:, :, :, aux_index:, aux_index:]

        x1 = x1 - p[:,:,0].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x0
        x2 = x2 - p[:,:,1].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x0
        x3 = x3 - p[:,:,2].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x0 - p[:,:,3].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x0_2

        # ---------------- update ----------------------
        x1_2 = torch.zeros(x0.shape,device=torch.device("cuda"))
        x2_2 = torch.zeros(x0.shape,device=torch.device("cuda"))

        x1_2[:, :, :, :-aux_index, :] = x1[:, :, :, aux_index:, :]
        x2_2[:, :, :, :, :-aux_index] = x2[:, :, :, :, aux_index:]

        x0 = x0 + u[:,:,0].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x1 + u[:,:,1].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x1_2
        x0 = x0 + u[:,:,2].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x2 + u[:,:,3].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x2_2
        x0 = x0 + u[:,:,4].unsqueeze(dim=0).unsqueeze(dim=3).unsqueeze(dim=4) * x3

        # ---------------- output ----------------------
        result = x0[:, :, :, :-aux_index, :-aux_index]

        return result

    def lifting_scheme_3d(self, x):
        """
        Perform the lifting scheme on each channel 
        """
        p = self.extract_param_p()
        u = self.extract_param_u()

        n, channel, h, w = x.shape

        result = []
        channel_split_num = self.channel_split
        split_channels = int(self.out_channels/channel_split_num)

        split_channel_result = torch.zeros([n, split_channels, channel, h, w],device=torch.device("cuda"))

        x = x.repeat((1,split_channels,1,1))
        x = x.reshape(n,split_channels,channel,h,w)

        for i in range(channel_split_num):
            split_channel_result = self.lifting_scheme_2d(x,p[i*split_channels:(i+1)*split_channels,:,:],u[i*split_channels:(i+1)*split_channels,:,:])
            split_channel_result = split_channel_result.sum(dim=2)
            result.append(split_channel_result)
        
        result = torch.cat(result, dim=1)
        return result
    
    def forward(self, x):
        pad = nn.ConstantPad2d((self.padding, self.padding, self.padding, self.padding), 0)
        x = pad(x)
        return self.lifting_scheme_3d(x)

