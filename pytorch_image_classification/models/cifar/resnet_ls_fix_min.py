'''
HW: replace all the strided conv by LSMP_AlterChan

LSML
Parameters:
    pumode: fix
    lsptype: min
    porder: one_low
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer
from ..functions.ls_quincunx import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, csv_file, downsample=None):
        super().__init__()

        if stride == 2:
            filename = csv_file + '/' + str(in_channels) + '_' + str(out_channels)
            self.conv1 = LSMP_AlterChan(in_chans =  in_channels, 
                                        out_chans=  out_channels, 
                                                    lsptype="min", 
                                                    pu_mode="fix", 
                                                    csv_file=filename,
                                                    porder = "one_low")
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,  
                bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += identity
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, csv_file, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        if stride == 2:
            filename = csv_file + '/' + str(out_channels) + '_' + str(out_channels)
            self.conv2 = LSMP_AlterChan(in_chans =  out_channels, 
                                        out_chans=  out_channels, 
                                                    lsptype="min", 
                                                    pu_mode="fix", 
                                                    csv_file=filename,
                                                    porder = "one_low") 
        else:
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,  # downsample with 3x3 conv
                padding=1,
                bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels,
                               out_channels * self.expansion,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = F.relu(self.bn2(self.conv2(y)), inplace=True)
        y = self.bn3(self.conv3(y))  # not apply ReLU
        y += identity
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_config = config.model.resnet_ls_fix_min
        n_blocks = model_config.n_blocks
        initial_channels = model_config.initial_channels
        block_type = model_config.block_type
        out_dir = config.train.output_dir

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = BottleneckBlock

        n_channels = [
            initial_channels,
            initial_channels * 2,
            initial_channels * 4,
            initial_channels * 8
        ]

        self.in_channel = initial_channels

        self.conv = nn.Conv2d(config.dataset.n_channels,
                              n_channels[0],
                              kernel_size=7,
                              stride=2,
                              padding=3,
                              bias=False)
        
        self.bn = nn.BatchNorm2d(initial_channels)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_blocks[0],
                                       block,
                                       stride=1,
                                       csv_file=out_dir)
        self.stage2 = self._make_stage(n_channels[1],
                                       n_blocks[1],
                                       block,
                                       stride=2,
                                       csv_file=out_dir)
        self.stage3 = self._make_stage(n_channels[2],
                                       n_blocks[2],
                                       block,
                                       stride=2,
                                       csv_file=out_dir)
        self.stage4 = self._make_stage(n_channels[3],
                                       n_blocks[3],
                                       block,
                                       stride=2,
                                       csv_file=out_dir)

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, config.dataset.n_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)

    def _make_stage(self, channel, n_blocks, block, stride, csv_file):
        stage = nn.Sequential()
        downsample = None
        if stride !=1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential()
            if stride == 2:
                filename = csv_file + '/' + str(self.in_channel) + '_' + str(channel * block.expansion) + "_2"
                downsample.add_module(
                    'lsmp',
                    LSMP_AlterChan( in_chans =  self.in_channel, 
                                    out_chans=  channel * block.expansion, 
                                    lsptype="min", 
                                    pu_mode="fix", 
                                    csv_file=filename,
                                    porder = "one_low"))
            else:
                downsample.add_module(
                    'conv',
                    nn.Conv2d(
                        self.in_channel,
                        channel * block.expansion,
                        kernel_size=1,
                        stride=stride,  # downsample
                        padding=0,
                        bias=False))
            downsample.add_module('bn', nn.BatchNorm2d(channel * block.expansion))  # BN
        
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(self.in_channel, channel,
                                      stride=stride, csv_file=csv_file,downsample=downsample))
                self.in_channel = channel * block.expansion
            else:
                stage.add_module(block_name,
                                 block(self.in_channel, channel, stride=1, csv_file=csv_file))
        return stage

    def _forward_conv(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
