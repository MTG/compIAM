
"""
Created on July 05, 2019

@author: Stefan Lattner

Sony CSL Paris, France

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Complex(nn.Module):
    def __init__(self, n_in, n_out, dropout=.5, learn_norm=False):
        super(Complex, self).__init__()

        self.layer = nn.Linear(n_in, n_out*2, bias=False)

        self.drop = nn.Dropout(dropout)
        self.learn_norm = learn_norm
        self.n_out = n_out
        self.norm_val = nn.Parameter(torch.Tensor([.43])) # any start value

    def drop_gauss(self, x):
        return torch.normal(mean=x, std=0.5)

    def forward(self, x):
        out = torch.matmul(self.drop(x), self.set_to_norm_graph(
            self.norm_val).transpose(0,1))
        real = out[:, :self.n_out]
        imag = out[:, self.n_out:]
        amplitudes = (real**2 + imag**2)**.5
        phases = torch.atan2(real, imag)
        return amplitudes, phases

    def backward(self, amplitudes, phases):
        real = torch.sin(phases) * amplitudes
        imag = torch.cos(phases) * amplitudes
        cat_ = torch.cat((real, imag), dim=1)
        recon = torch.matmul(cat_, self.set_to_norm_graph(self.norm_val))
        return recon

    def set_to_norm(self, val):
        """
        Sets the norms of all convolutional kernels of the C-GAE to a specific
        value.

        :param val: norms of kernels are set to this value
        """
        if val == -1:
            val = self.norm_val
        shape_x = self.layer.weight.size()
        conv_x_reshape = self.layer.weight.view(shape_x[0], -1)
        norms_x = ((conv_x_reshape ** 2).sum(1) ** .5).view(-1, 1)
        conv_x_reshape = conv_x_reshape / norms_x
        weight_x_new = (conv_x_reshape.view(*shape_x) * val).clone()
        self.layer.weight.data = weight_x_new

    def set_to_norm_graph(self, val):
        if not self.learn_norm:
            return self.layer.weight
        """
        Sets the norms of all convolutional kernels of the C-GAE to a learned
        value.

        :param val: norms of kernels are set to this value
        """
        if val == -1:
            val = self.norm_val
        shape_x = self.layer.weight.size()
        conv_x_reshape = self.layer.weight.view(shape_x[0], -1)
        norms_x = ((conv_x_reshape ** 2).sum(1) ** .5).view(-1, 1)
        conv_x_reshape = conv_x_reshape / norms_x
        weight_x_new = (conv_x_reshape.view(*shape_x) * val).clone()
        return weight_x_new
