''' 
Lifting Scheme-Based Morphological Layer in PyTorch ----- LSML
Lifting Scheme-Based Frequency Layer in PyTorch --------- LSFL
'''
# 
# LSMP: Lifting Scheme-based Morphological Primitive 
# LSMP_AlterChan: LSMP + channel alternation module
# 
# LSML: LSMP(in_channel, ptype, pu_mode, csv_file, porder)
# Optional parameters:
#   porder = one_low
#       pu_mode = fix or learn
#       ptype   = max or min or smooth or attention
#
# LSFL: LSMP(in_channel, ptype, pu_mode, csv_file, porder)
# Optional parameters:
#   porder = one_fusion_HL or one_fusion_LH or one_fusion_HH or two_tree_HL or two_tree_LH or two_tree_HH
#       pu_mode = fix or learn
#       ptype   = max or min or smooth or attention


import torch
import torch.nn as nn
from ..functions.utils import *


# =========================== util functions =======================
def obtain_verhor(x, i1, i2, step1, j1, j2, step2):
    a = x[:, :, i1-1:i2-1:step1, j1:j2:step2]   # up
    b = x[:, :, i1:i2:step1, j1-1:j2-1:step2]   # left
    c = x[:, :, i1+1:i2+1:step1, j1:j2:step2]   # down
    d = x[:, :, i1:i2:step1, j1+1:j2+1:step2]   # right
    return a, b, c, d

def obtain_diag(x, i1, i2, step1, j1, j2, step2):
    a = x[:, :, i1-1:i2-1:step1, j1-1:j2-1:step2]   # left-up
    b = x[:, :, i1+1:i2+1:step1, j1-1:j2-1:step2]   # left-down
    c = x[:, :, i1+1:i2+1:step1, j1+1:j2+1:step2]   # right-down
    d = x[:, :, i1-1:i2-1:step1, j1+1:j2+1:step2]   # right-up
    return a, b, c, d


# =========================== LSMP =================================
class LSMP(nn.Module):
    def __init__(self, in_channel,ptype="max", pu_mode="fix", csv_file="save", porder = "one_low"):
        super(LSMP, self).__init__()
        self.pad = nn.ConstantPad2d(1, 0)

        self.ptype = ptype
        if ptype not in ["max", "min", "smooth", "attention"]:
            raise ValueError("Type of LSMP must be 'max' , 'min' , 'smooth' or 'attention' !")

        self.pumode = pu_mode
        if self.pumode not in ["fix", "learn"]:
            raise ValueError("P/U mode must be 'fix' or 'learn'!")
        
        self.porder = porder
        if porder not in ["one_low", "one_fusion_LH", "one_fusion_HL", "one_fusion_HH", "one_fusion_all", "two_tree_HH", "two_tree_HL", "two_tree_LH", "two_tree_all"]:
            raise ValueError("Type of LSMP must be 'one_low' , 'one_fusion_LH' , 'one_fusion_HL' , 'one_fusion_HH' , 'one_fusion_all' , 'two_tree_LH' or 'two_tree_HL' or 'two_tree_HH' or 'two_tree_all' !")
        
        if self.porder == "one_fusion_LH" or "one_fusion_HL" or "one_fusion_HH":
            self.onefusion = nn.Parameter(torch.rand(1))
        if self.porder == "one_fusion_all":
            self.lhfusion = nn.Parameter(torch.rand(1))
            self.hlfusion = nn.Parameter(torch.rand(1))
            self.hhfusion = nn.Parameter(torch.rand(1))
        if self.porder == "two_tree_LH" or "two_tree_HL" or "two_tree_HH" or "two_tree_all":
            self.rd_low  = nn.Parameter(torch.rand(1))
            self.rd_high = nn.Parameter(torch.rand(1))
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

        if self.ptype == "attention":
            # weights for vertical/horizontal lifting
            self.conv_vh1 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)
            self.conv_vh2 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)
            self.conv_vh3 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)
            self.conv_vh4 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)

            # weights for diagonal lifting
            self.conv_di1 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)
            self.conv_di2 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)
            self.conv_di3 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)
            self.conv_di4 = nn.Conv2d(in_channel, in_channel, 1, groups=in_channel)

        if self.pumode == "fix":
            self.pweight = 1
            self.uweight = 0.5
        
        else:
            self.pweight = nn.Parameter(torch.rand(1))#torch.ones(1))
            self.uweight = nn.Parameter(torch.rand(1))#torch.ones(1))

            self.csv_file_pu = csv_file + "_puweights.csv"
            head = ["pweight", "uweight"]
            create_csv(self.csv_file_pu, head)

        if self.porder == 'one_low':
            self.csv_file_fusion = "None"
        elif self.porder == 'one_fusion_LH':
            self.csv_file_fusion = csv_file + "_LHfusion.csv"
            head = ["LH_fusion_parameter"]
            create_csv(self.csv_file_fusion, head)
        elif self.porder == 'one_fusion_HL':
            self.csv_file_fusion = csv_file + "_HLfusion.csv"
            head = ["HL_fusion_parameter"]
            create_csv(self.csv_file_fusion, head)
        elif self.porder == 'one_fusion_HH':
            self.csv_file_fusion = csv_file + "_HHfusion.csv"
            head = ["HH_fusion_parameter"]
            create_csv(self.csv_file_fusion, head)
        elif self.porder == "one_fusion_all":
            self.csv_file_fusion = csv_file + "_allfusion.csv"
            head = ["LH_fusion_parameter", "HL_fusion_parameter", "HH_fusion_parameter"]
            create_csv(self.csv_file_fusion, head)
        else:
            self.csv_file_fusion = csv_file + "_rd_params.csv"
            head = ["low_parameter", "high_parameter"]
            create_csv(self.csv_file_fusion, head)

        print("LMSP(",self.ptype,",", self.pumode, "," ,self.porder, ") is used!")

    def forward(self, x):
        x = self.pad(x)
        _, _, h, w = x.shape

        # define x_LL,h_LL,w_LL
        x_LL = x[:, :, 1:h-2:2, 1:w-2:2]
        x_LL = self.pad(x_LL)
        x_high = x_LL
        _, _, h_LL, w_LL = x_LL.shape

        pweight = abs(self.pweight)
        uweight = abs(self.uweight)

        if self.ptype == 'max' or 'min' or 'smooth':
            # vertical/horizontal lifting
            x[:, :, 1:h-2:2, 2:w-1:2] -= pweight * self.qua_verhor(x, 1, h-2, 2, 2, w-1, 2)
            x[:, :, 2:h-1:2, 1:w-2:2] -= pweight * self.qua_verhor(x, 2, h-1, 2, 1, w-2, 2)
            x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.qua_verhor(x, 1, h-2, 2, 1, w-2, 2)
            x[:, :, 2:h-1:2, 2:w-1:2] += uweight * self.qua_verhor(x, 2, h-1, 2, 2, w-1, 2)

            if self.porder == "one_low":
                # diagonal lifting of red
                x[:, :, 2:h-1:2, 2:w-1:2] -= pweight * self.qua_diag(x, 2, h-1, 2, 2, w-1, 2)
                x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.qua_diag(x, 1, h-2, 2, 1, w-2, 2)
            elif self.porder == "one_fusion_LH" or "one_fusion_HL" or "one_fusion_HH" or "one_fusion_all":
                # diagonal lifting of red
                x[:, :, 2:h-1:2, 2:w-1:2] -= pweight * self.qua_diag(x, 2, h-1, 2, 2, w-1, 2)   #LH
                x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.qua_diag(x, 1, h-2, 2, 1, w-2, 2)   #LL
                # diagonal lifting of black
                x[:, :, 2:h-1:2, 1:w-2:2] -= pweight * self.qua_diag(x, 2, h-1, 2, 1, w-2, 2)   #HH
                x[:, :, 1:h-2:2, 2:w-1:2] += uweight * self.qua_diag(x, 1, h-2, 2, 2, w-1, 2)   #HL
            else:
                # diagonal lifting of red
                x[:, :, 2:h-1:2, 2:w-1:2] -= pweight * self.qua_diag(x, 2, h-1, 2, 2, w-1, 2)   #LH
                x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.qua_diag(x, 1, h-2, 2, 1, w-2, 2)   #LL
                # diagonal lifting of black
                x[:, :, 2:h-1:2, 1:w-2:2] -= pweight * self.qua_diag(x, 2, h-1, 2, 1, w-2, 2)   #HH
                x[:, :, 1:h-2:2, 2:w-1:2] += uweight * self.qua_diag(x, 1, h-2, 2, 2, w-1, 2)   #HL

                # the second order decomposition
                x_LL = x[:, :, 1:h-2:2, 1:w-2:2]
                x_LL = self.pad(x_LL)
                _, _, h_LL, w_LL = x_LL.shape
                # vertical/horizontal lifting
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] -= pweight * self.qua_verhor(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] -= pweight * self.qua_verhor(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] += uweight * self.qua_verhor(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)
                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] += uweight * self.qua_verhor(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)
                # diagonal lifting of red
                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] -= pweight * self.qua_diag(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)   #LH
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] += uweight * self.qua_diag(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)   #LL
                # diagonal lifting of black
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] -= pweight * self.qua_diag(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)   #HH
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] += uweight * self.qua_diag(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)   #HL

                # wavelet tree select
                if self.porder == "two_tree_LH":    
                    x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] = 0  #HH
                    x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] = 0  #HL
                if self.porder == "two_tree_HL":
                    x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] = 0  #HH
                    x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] = 0  #LH
                if self.porder == "two_tree_HH":
                    x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] = 0  #HL
                    x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] = 0  #LH
                if self.porder == "two_tree_all":
                    x_high = x_LL
                    x_high[:, :, 1:h_LL-2:2, 1:w_LL-2:2] = 0  #LL

                    x_high[:, :, 1:h_LL-2:2, 2:w_LL-1:2] -= uweight * self.qua_diag(x_high, 1, h_LL-2, 2, 2, w_LL-1, 2)   #HL
                    x_high[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.qua_diag(x_high, 2, h_LL-1, 2, 1, w_LL-2, 2)   #HH
                    x_high[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.qua_diag(x_high, 1, h_LL-2, 2, 1, w_LL-2, 2)   #LL
                    x_high[:, :, 2:h_LL-1:2, 2:w_LL-1:2] += pweight * self.qua_diag(x_high, 2, h_LL-1, 2, 2, w_LL-1, 2)   #LH

                    x_high[:, :, 2:h_LL-1:2, 2:w_LL-1:2] -= uweight * self.qua_verhor(x_high, 2, h_LL-1, 2, 2, w_LL-1, 2)
                    x_high[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.qua_verhor(x_high, 1, h_LL-2, 2, 1, w_LL-2, 2)
                    x_high[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.qua_verhor(x_high, 2, h_LL-1, 2, 1, w_LL-2, 2)
                    x_high[:, :, 1:h_LL-2:2, 2:w_LL-1:2] += pweight * self.qua_verhor(x_high, 1, h_LL-2, 2, 2, w_LL-1, 2)

                    
                    x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] = 0 #HH
                    x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] = 0 #HL
                    x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] = 0 #LH

                # lifting scheme red-black reverse transform:  (h/4, w/4) --> (h/2, w/2)
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] -= uweight * self.qua_diag(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)   #HL
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.qua_diag(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)   #HH
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.qua_diag(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)   #LL
                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] += pweight * self.qua_diag(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)   #LH

                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] -= uweight * self.qua_verhor(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.qua_verhor(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.qua_verhor(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] += pweight * self.qua_verhor(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)

        else:
            # vertical/horizontal lifting
            x[:, :, 1:h-2:2, 2:w-1:2] -= pweight * self.adapt_qua_verhor(x, 1, h-2, 2, 2, w-1, 2)
            x[:, :, 2:h-1:2, 1:w-2:2] -= pweight * self.adapt_qua_verhor(x, 2, h-1, 2, 1, w-2, 2)
            x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.adapt_qua_verhor(x, 1, h-2, 2, 1, w-2, 2)
            x[:, :, 2:h-1:2, 2:w-1:2] += uweight * self.adapt_qua_verhor(x, 2, h-1, 2, 2, w-1, 2)

            if self.porder == "one_low":
                # diagonal lifting of red
                x[:, :, 2:h-1:2, 2:w-1:2] -= pweight * self.adapt_qua_diag(x, 2, h-1, 2, 2, w-1, 2)
                x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.adapt_qua_diag(x, 1, h-2, 2, 1, w-2, 2)
            elif self.porder == "one_fusion_LH" or "one_fusion_HL" or "one_fusion_HH" or "one_fusion_all":
                # diagonal lifting of red
                x[:, :, 2:h-1:2, 2:w-1:2] -= pweight * self.adapt_qua_diag(x, 2, h-1, 2, 2, w-1, 2)   #LH
                x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.adapt_qua_diag(x, 1, h-2, 2, 1, w-2, 2)   #LL
                # diagonal lifting of black
                x[:, :, 2:h-1:2, 1:w-2:2] -= pweight * self.adapt_qua_diag(x, 2, h-1, 2, 1, w-2, 2)   #HH
                x[:, :, 1:h-2:2, 2:w-1:2] += uweight * self.adapt_qua_diag(x, 1, h-2, 2, 2, w-1, 2)   #HL
            else:
                # diagonal lifting of red
                x[:, :, 2:h-1:2, 2:w-1:2] -= pweight * self.adapt_qua_diag(x, 2, h-1, 2, 2, w-1, 2)   #LH
                x[:, :, 1:h-2:2, 1:w-2:2] += uweight * self.adapt_qua_diag(x, 1, h-2, 2, 1, w-2, 2)   #LL
                # diagonal lifting of black
                x[:, :, 2:h-1:2, 1:w-2:2] -= pweight * self.adapt_qua_diag(x, 2, h-1, 2, 1, w-2, 2)   #HH
                x[:, :, 1:h-2:2, 2:w-1:2] += uweight * self.adapt_qua_diag(x, 1, h-2, 2, 2, w-1, 2)   #HL

                # the second order decomposition
                x_LL = x[:, :, 1:h-2:2, 1:w-2:2]
                x_LL = self.pad(x_LL)
                _, _, h_LL, w_LL = x_LL.shape
                # vertical/horizontal lifting
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] -= pweight * self.adapt_qua_verhor(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] -= pweight * self.adapt_qua_verhor(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] += uweight * self.adapt_qua_verhor(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)
                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] += uweight * self.adapt_qua_verhor(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)
                # diagonal lifting of red
                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] -= pweight * self.adapt_qua_diag(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)   #LH
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] += uweight * self.adapt_qua_diag(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)   #LL
                # diagonal lifting of black
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] -= pweight * self.adapt_qua_diag(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)   #HH
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] += uweight * self.adapt_qua_diag(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)   #HL

                # wavelet tree select
                if self.porder == "two_tree_LH":    
                    x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] = 0  #HH
                    x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] = 0  #HL
                if self.porder == "two_tree_HL":
                    x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] = 0  #HH
                    x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] = 0  #LH
                if self.porder == "two_tree_HH":
                    x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] = 0  #HL
                    x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] = 0  #LH
                if self.porder == "two_tree_all":
                    # using HL2,LH2,HH2 to reconstruct the detail of LL1 (x_high: (h/2, w/2))
                    x_high = x_LL
                    x_high[:, :, 1:h_LL-2:2, 1:w_LL-2:2] = 0  #LL

                    x_high[:, :, 1:h_LL-2:2, 2:w_LL-1:2] -= uweight * self.adapt_qua_diag(x_high, 1, h_LL-2, 2, 2, w_LL-1, 2)   #HL
                    x_high[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.adapt_qua_diag(x_high, 2, h_LL-1, 2, 1, w_LL-2, 2)   #HH
                    x_high[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.adapt_qua_diag(x_high, 1, h_LL-2, 2, 1, w_LL-2, 2)   #LL
                    x_high[:, :, 2:h_LL-1:2, 2:w_LL-1:2] += pweight * self.adapt_qua_diag(x_high, 2, h_LL-1, 2, 2, w_LL-1, 2)   #LH

                    x_high[:, :, 2:h_LL-1:2, 2:w_LL-1:2] -= uweight * self.adapt_qua_verhor(x_high, 2, h_LL-1, 2, 2, w_LL-1, 2)
                    x_high[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.adapt_qua_verhor(x_high, 1, h_LL-2, 2, 1, w_LL-2, 2)
                    x_high[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.adapt_qua_verhor(x_high, 2, h_LL-1, 2, 1, w_LL-2, 2)
                    x_high[:, :, 1:h_LL-2:2, 2:w_LL-1:2] += pweight * self.adapt_qua_verhor(x_high, 1, h_LL-2, 2, 2, w_LL-1, 2)

                    # using LL2 to reconstruct the approximate of LL1 (x_LL: (h/2, w/2))
                    x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] = 0 #HH
                    x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] = 0 #HL
                    x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] = 0 #LH

                # lifting scheme red-black reverse transform (h/4, w/4) --> (h/2, w/2)
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] -= uweight * self.adapt_qua_diag(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)   #HL
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.adapt_qua_diag(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)   #HH
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.adapt_qua_diag(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)   #LL
                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] += pweight * self.adapt_qua_diag(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)   #LH

                x_LL[:, :, 2:h_LL-1:2, 2:w_LL-1:2] -= uweight * self.adapt_qua_verhor(x_LL, 2, h_LL-1, 2, 2, w_LL-1, 2)
                x_LL[:, :, 1:h_LL-2:2, 1:w_LL-2:2] -= uweight * self.adapt_qua_verhor(x_LL, 1, h_LL-2, 2, 1, w_LL-2, 2)
                x_LL[:, :, 2:h_LL-1:2, 1:w_LL-2:2] += pweight * self.adapt_qua_verhor(x_LL, 2, h_LL-1, 2, 1, w_LL-2, 2)
                x_LL[:, :, 1:h_LL-2:2, 2:w_LL-1:2] += pweight * self.adapt_qua_verhor(x_LL, 1, h_LL-2, 2, 2, w_LL-1, 2)

        if self.pumode == "learn":
            content = [pweight, uweight]
            append_csv(self.csv_file_pu, content)
        if self.porder == "one_fusion_LH" or "one_fusion_HL" or "one_fusion_HH":
            content = [self.onefusion]
            append_csv(self.csv_file_fusion, content)
        elif self.porder == "one_fusion_all":
            content = [self.lhfusion, self.hlfusion, self.hhfusion]
            append_csv(self.csv_file_fusion, content)
        else:
            content = [self.rd_low, self.rd_high]
            append_csv(self.csv_file_fusion, content)

        if self.porder == "one_low":
            return x[:, :, 1:h-2:2, 1:w-2:2].contiguous()
        elif self.porder == "one_fusion_LH" :
            x_fusion = x[:, :, 1:h-2:2, 1:w-2:2] + self.onefusion * x[:, :, 2:h-1:2, 2:w-1:2]   # one_fusion_LH
            return x_fusion.contiguous()
        elif self.porder == "one_fusion_HL":
            x_fusion = x[:, :, 1:h-2:2, 1:w-2:2] + self.onefusion * x[:, :, 1:h-2:2, 2:w-1:2]   # one_fusion_HL
            return x_fusion.contiguous()
        elif self.porder == "one_fusion_HH":
            x_fusion = x[:, :, 1:h-2:2, 1:w-2:2] + self.onefusion * x[:, :, 2:h-1:2, 1:w-2:2]   # one_fusion_HH
            return x_fusion.contiguous()
        elif self.porder == "one_fusion_all":
            x_fusion = x[:, :, 1:h-2:2, 1:w-2:2] + self.lhfusion * x[:, :, 2:h-1:2, 2:w-1:2] + self.hlfusion * x[:, :, 1:h-2:2, 2:w-1:2] + self.hhfusion * x[:, :, 2:h-1:2, 1:w-2:2]
            return x_fusion.contiguous()
        elif self.porder == "two_tree_LH":
            x_fusion = self.rd_low * x_LL[:, :, 1:h_LL-1, 1:w_LL-1] + self.rd_high * x[:, :, 2:h-1:2, 2:w-1:2]
            return x_fusion
        elif self.porder == "two_tree_HL":
            x_fusion = self.rd_low * x_LL[:, :, 1:h_LL-1, 1:w_LL-1] + self.rd_high * x[:, :, 1:h-2:2, 2:w-1:2]
            return x_fusion
        elif self.porder == "two_tree_HH":
            x_fusion = self.rd_low * x_LL[:, :, 1:h_LL-1, 1:w_LL-1] + self.rd_high * x[:, :, 2:h-1:2, 1:w-2:2]
            return x_fusion
        else:
            x_fusion = self.rd_low * x_LL[:, :, 1:h_LL-1, 1:w_LL-1] + self.rd_high * x_high[:, :, 1:h_LL-1, 1:w_LL-1]
            return x_fusion
    
    def adapt_qua_verhor(self, x, i1, i2, step1, j1, j2, step2):
        a, b, c, d = obtain_verhor(x, i1, i2, step1, j1, j2, step2)
        a = self.conv_vh1(a)
        b = self.conv_vh2(b)
        c = self.conv_vh3(c)
        d = self.conv_vh4(d)
        value = a + b + c + d
        return value

    def adapt_qua_diag(self, x, i1, i2, step1, j1, j2, step2):
        a, b, c, d = obtain_diag(x, i1, i2, step1, j1, j2, step2)
        a = self.conv_di1(a)
        b = self.conv_di2(b)
        c = self.conv_di3(c)
        d = self.conv_di4(d)
        value = a + b + c + d
        return value
    
    def qua_verhor(self, x, i1, i2, step1, j1, j2, step2):
        a, b, c, d = obtain_verhor(x, i1, i2, step1, j1, j2, step2)
        if self.ptype == "max":
            value, _ = torch.max(torch.stack((a, b, c, d)).detach(), 0)
        elif self.ptype == "min":
            value, _ = torch.min(torch.stack((a, b, c, d)).detach(), 0)
        else:
            value = torch.mean(torch.stack((a, b, c, d)).detach(), 0)
        return value

    def qua_diag(self, x, i1, i2, step1, j1, j2, step2):
        a, b, c, d = obtain_diag(x, i1, i2, step1, j1, j2, step2)
        if self.ptype == "max":
            value, _ = torch.max(torch.stack((a, b, c, d)).detach(), 0)
        elif self.ptype == "min":
            value, _ = torch.min(torch.stack((a, b, c, d)).detach(), 0)
        else:
            value = torch.mean(torch.stack((a, b, c, d)).detach(), 0)
        return value

#--------------LSMP_AlterChan--------------------
class LSMP_AlterChan(nn.Module):
    def __init__(self, in_chans, out_chans, lsptype="max", pu_mode="fix", csv_file="save", porder = "one_low"):
        super(LSMP_AlterChan, self).__init__()
        self.fdsp = LSMP(in_channel=in_chans,ptype=lsptype, pu_mode=pu_mode, csv_file=csv_file, porder=porder)
        self.alter_chan = nn.Conv2d(in_chans, out_chans, 1, bias=False)
    
    def forward(self, x):
        x = self.fdsp(x)
        x = self.alter_chan(x)
        return x