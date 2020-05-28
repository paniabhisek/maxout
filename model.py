#/usr/bin/python3
# -*- coding: utf-8 -*-

#####################################
#  https://arxiv.org/abs/1302.4389  #
#####################################

# Library modules
import json

# External library modules
import torch

# local library modules
from maxout import MaxoutMLP
from maxout import MaxoutConv
from utils import init_hyper_params
from utils import num_corrects
from utils import device

class MaxoutMLPMNIST(torch.nn.Module):
    """MLP + dropout"""

    def __init__(self, input_dim=784):
        """
        Define maxout layers to train MNIST dataset

        :param input_dim: Input dimension to the model
                          For mnist case :py:const:28: `x` :py:const:28: dimension
                          images.
                          For this model `2d` convolution is reshaped
                          to create `1d` pixels.
        :type input_dim: :py:obj:`int`
        """
        super(MaxoutMLPMNIST, self).__init__()

        # parameters initialization
        self.hparams = init_hyper_params()

        # dummy tensors for upper bound after norm
        self.norm_upper1 = torch.empty(self.hparams['mlp'][0]['neurons']).\
                           fill_(self.hparams['norm_constraint']).to(device)

        self.norm_upper2 = torch.empty(self.hparams['mlp'][1]['neurons']).\
                           fill_(self.hparams['norm_constraint']).to(device)

        # Maxout Layer 1 (input_size, num_layers, num_neurons)
        self.maxout1 = MaxoutMLP(input_dim,
                                 self.hparams['mlp'][0]['layers'],
                                 self.hparams['mlp'][0]['neurons']).to(device)

        # Maxout Layer 2 (input_size, num_layers, num_neurons)
        self.maxout2 = MaxoutMLP(self.hparams['mlp'][0]['neurons'],
                                 self.hparams['mlp'][1]['layers'],
                                 self.hparams['mlp'][1]['neurons']).to(device)

    def forward(self, input_imgs, is_train=True):
        """
        Function to forward inputs to the maxout network

        The norms and it's constraint are only added for MNIST dataset
        as described in paper.

        :param input_imgs: Input images to the model
        :type input_imgs: :py:class:`torch.Tensor`
        :param is_train: Whether the input is for training.
        :type is_train: :py:obj:`bool`
        """
        dropout = torch.nn.Dropout(p=0.2).to(device)

        # Maxout layer1 + dropout
        mx1_out = self.maxout1(input_imgs, is_norm=True,
                              norm_constraint=self.hparams['norm_constraint'],
                              norm_upper=self.norm_upper1)
        if is_train:
            mx1_out = dropout(mx1_out)

        # Maxout layer2 + dropout
        mx2_out = self.maxout2(mx1_out, is_norm=True,
                              norm_constraint=self.hparams['norm_constraint'],
                              norm_upper=self.norm_upper2)

        # Softmax layer
        softmax = torch.nn.Softmax()
        softmax = softmax(mx2_out)

        return softmax

class MaxoutConvMNIST(torch.nn.Module):
    def __init__(self, in_channels=1):
        """
        Define Maxout, Maxpool and Linear Layers for the model

        :param in_channels: Number of channel of input image
        :type in_channels: :py:obj:`int`
        """
        super(MaxoutConvMNIST, self).__init__()

        # parameters initialization
        self.hparams = init_hyper_params()

        # Maxout Layer 1 (in_channels, out_channels, kernel)
        self.maxout1 = MaxoutConv(in_channels=in_channels,
                                  out_channels=self.hparams['conv'][0]['channels'],
                                  kernel_size=self.hparams['conv'][0]['kernel'],
                                  padding=self.hparams['padding'][0]).to(device)
        self.maxpool1 = torch.nn.MaxPool2d(self.hparams['pool'][0],
                                           self.hparams['stride'][0])

        # Maxout Layer 2 (in_channels, out_channels, kernel)
        self.maxout2 = MaxoutConv(in_channels=1,
                                  out_channels=self.hparams['conv'][1]['channels'],
                                  kernel_size=self.hparams['conv'][1]['kernel'],
                                  padding=self.hparams['padding'][1]).to(device)
        self.maxpool2 = torch.nn.MaxPool2d(self.hparams['pool'][1],
                                           self.hparams['stride'][1])

        # Maxout Layer 3 (in_channels, out_channels, kernel)
        self.maxout3 = MaxoutConv(in_channels=1,
                                  out_channels=self.hparams['conv'][2]['channels'],
                                  kernel_size=self.hparams['conv'][2]['kernel'],
                                  padding=self.hparams['padding'][2]).to(device)
        self.maxpool3 = torch.nn.MaxPool2d(self.hparams['pool'][2],
                                           self.hparams['stride'][2])

        self.linear = torch.nn.Linear(self.hparams['linear']['in_channels'],
                                      self.hparams['linear']['out_channels'])

    def forward(self, _input, is_train=True):
        """
        Pass the input to the whole model

        :param _input: input image
        :type _input: :py:class:`torch.Tensor`
        """
        # Maxout1 + Maxpool1
        out = self.maxout1(_input, is_norm=True)
        out = self.maxpool1(out)

        # Maxout2 + Maxpool2
        out = self.maxout2(out, is_norm=True)
        out = self.maxpool2(out)

        # Maxout3 + Maxpool3
        out = self.maxout3(out, is_norm=True)
        out = self.maxpool3(out)

        out = out.view(out.shape[0], -1)

        # linear
        out = self.linear(out)

        return torch.nn.Softmax()(out)
