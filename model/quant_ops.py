#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

import collections
import math
from os import stat
import pdb
import random
import time
from itertools import repeat
# from turtle import forward
# from turtle import forward
from model.binarize import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from importlib import import_module

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def quant_max(tensor):
    """
    Returns the max value for symmetric quantization.
    """
    return torch.abs(tensor.detach()).max() + 1e-8

def TorchRound():
    """
    Apply STE to clamp function.
    """
    class identity_quant(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = torch.round(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

    return identity_quant().apply

class quant_weight(nn.Module):
    """
    Quantization function for quantize weight with maximum.
    """

    def __init__(self, k_bits):
        super(quant_weight, self).__init__()
        self.k_bits = k_bits
        self.qmax = 2. ** (k_bits -1) - 1.
        self.round = TorchRound()

    def forward(self, input):

        max_val = quant_max(input)
        weight = input * self.qmax / max_val
        q_weight = self.round(weight)
        q_weight = q_weight * max_val / self.qmax
        return q_weight
def uniform_quantize(k):
  class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

class dorefa_quant_weight(nn.Module):
    def __init__(self, k_bits):
        super(dorefa_quant_weight, self).__init__()
        self.k_bits = k_bits
        self.round = uniform_quantize(k_bits)
    
    def forward(self, input):
        if self.k_bits == 32:
            return input
        else:
            w = torch.tanh(input)
            max_w = max(torch.abs(w)).detach()
            w = w/(2*max_w) + 0.5
            qw = max_w * (2 * self.round(w)-1)
            return qw

class dorefa_quant_act(nn.Module):
    def __init__(self, k_bits):
        super(dorefa_quant_act, self).__init__()
        self.k_bits = k_bits
        self.round = uniform_quantize(k_bits)
    
    def forward(self,input):
        if self.k_bits == 32:
            return input
        else:
            qa = self.round(torch.clamp(input,0,1))
            return qa

def pact_quantize(k):

    class qfn(torch.autograd.Function):
        
        # print("k"+str(k))
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.save_for_backward(x, alpha)
            if k == 32:
                out = x
            elif k == 1:
                out = torch.sign(x)
            else:
                y = torch.clamp(x, min = 0, max = alpha.item())
                n = (2**k-1)/alpha
                
                out = torch.round(y*n)/n
            return out

        @staticmethod
        def backward(ctx, gx):
            grad_input = gx.clone()
            x,alpha = ctx.saved_tensors
            lower = x < 0
            upper = x > alpha
            x_range = ~(lower|upper)
            ga = torch.sum(gx * torch.ge(x,alpha).float()).view(-1)
            return grad_input*x_range.float(), ga

    return qfn().apply

class pact_quant_act(nn.Module):
    def __init__(self, k_bits):
        super(pact_quant_act, self).__init__()
        self.k_bits = k_bits
        self.alpha = nn.Parameter(torch.tensor(10.))
        self.round = pact_quantize(k_bits)
    
    def forward(self,input):
        if self.k_bits == 32:
            return input
        else:
            qa = self.round(input,self.alpha)
            return qa

class quant_activation(nn.Module):
    """
    Quantization function for quantize activation with maximum and minimum, only for gate.
    """

    def __init__(self, k_bits=8):
        super(quant_activation, self).__init__()
        self.k_bits = k_bits
        self.round = TorchRound()

    def forward(self, input):

        act = input.detach()
        max_val, min_val = torch.max(act), torch.min(act)

        n = 2 ** self.k_bits - 1
        scale_factor = n / (max_val - min_val)
        zero_point = scale_factor * min_val

        zero_point = self.round(zero_point)
        zero_point += 2 ** (self.k_bits - 1)

        act = scale_factor * act - zero_point
        q_act = self.round(act)
        q_act = (q_act + zero_point) / scale_factor
        return q_act

class pams_quant_act(nn.Module):
    """
    Quantization function for quantize activation with parameterized max scale.
    """
    def __init__(self, k_bits, ema_epoch=1, decay=0.9997):
        super(pams_quant_act, self).__init__()
        self.decay = decay
        self.k_bits = k_bits
        self.qmax = 2. ** (self.k_bits -1) -1.
        self.round = TorchRound()
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.ema_epoch = ema_epoch
        self.epoch = 1
        self.iteration = 0
        # self.max_val = 0
        self.ema_scale = 1
        self.register_buffer('max_val', torch.zeros(1))
        self.error = 0
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.alpha, 10)

    def _ema(self, x):
        max_val = torch.mean(torch.max(torch.max(torch.max(abs(x),dim=1)[0],dim=1)[0],dim=1)[0])

        if self.epoch == 1:
            # print('aa')
            self.max_val = max_val
        else:
            # print('xx')
            self.max_val = (1.0-self.decay) * max_val + self.decay * self.max_val

    def forward(self, x):

        if self.epoch > self.ema_epoch or not self.training:
            act = torch.max(torch.min(x, self.alpha), -self.alpha)
        
        elif self.epoch <= self.ema_epoch and self.training:
            act = x
            self._ema(x)
            self.alpha.data = self.max_val.unsqueeze(0)
        # print(self.max_val)
        # # 3 bit
        # tmp_act = act * (2. ** (3 -1) -1.) / self.alpha
        # tmp_act = self.round(tmp_act)
        # tmp_act = tmp_act * self.alpha / (2. ** (3 -1) -1.)
        # qe3 = torch.mean((tmp_act.detach() - x.detach()) ** 2)
        #
        # # 2 bit
        # tmp_act = act * (2. ** (2 - 1) - 1.) / self.alpha
        # tmp_act = self.round(tmp_act)
        # tmp_act = tmp_act * self.alpha / (2. ** (2 - 1) - 1.)
        # qe2 = torch.mean((tmp_act.detach() - x.detach()) ** 2)
        #
        # # print(qe3, qe2)
        # if qe3 > qe2:
        #     import IPython
        #     IPython.embed()

        act = act * self.qmax / self.alpha
        q_act = self.round(act)
        q_act = q_act * self.alpha / self.qmax
        self.error = torch.mean((q_act.detach() - x.detach()) ** 2).item()


        # import IPython
        # IPython.embed()
        return q_act



class QuantConv2d_dofefa(nn.Module):
    """
    A convolution layer with quantized weight.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=32,):
        super(QuantConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quant_weight = dorefa_quant_weight(k_bits = k_bits)
        self.output = None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameter(self):
        stdv = 1.0/ math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias_flag:
            nn.init.constant_(self.bias,0.0)

    def forward(self, input, order=None):
        return nn.functional.conv2d(input, self.quant_weight(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)


class BinConv2d(nn.Conv2d):

    def __init__(self,**kwargs):
        super(BinConv2d,self).__init__(**kwargs)
        self.bw_func = binary_weight()
        self.ba_func = binary_activation()
        
    def forward(self,x):
        a = x
        w = self.weight

        ba = self.ba_func(a)
        bw = self.bw_func(w)
        
        return F.conv2d(ba,bw,self.bias,self.stride,self.padding,self.dilation,self.groups)

def bin_conv3x3(in_channels, out_channels,kernel_size=3,padding = 1,stride=1,bias = False):
     return BinConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride = stride,padding=padding,bias = bias)

if __name__ == '__main__':
    pact_quant_activation(8)
    pact_quant_activation(4)
    pact_quant_activation(3)
    pact_quant_activation(2)