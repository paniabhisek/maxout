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
from maxout import Maxout
from utils import *

class MaxoutMNIST(torch.nn.Module):
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
        super(MaxoutMNIST, self).__init__()

        # parameters initialization
        self.hparams = None
        self.init_hyper_params()

        # dummy tensors for upper bound after norm
        self.norm_upper1 = torch.empty(self.hparams['layer1']['linear.neurons']).\
                           fill_(self.hparams['norm_constraint']).to(device)

        self.norm_upper2 = torch.empty(self.hparams['layer2']['linear.neurons']).\
                           fill_(self.hparams['norm_constraint']).to(device)

        # Maxout Layer 1 (input_size, num_layers, num_neurons)
        self.maxout1 = Maxout(input_dim,
                              self.hparams['layer1']['linear.layers'],
                              self.hparams['layer1']['linear.neurons']).to(device)

        # Maxout Layer 2 (input_size, num_layers, num_neurons)
        self.maxout2 = Maxout(self.hparams['layer1']['linear.neurons'],
                              self.hparams['layer2']['linear.layers'],
                              self.hparams['layer2']['linear.neurons']).to(device)

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

    def init_hyper_params(self):
        """
        Initialize hyper parameters.

        Stores number of neurons in each layer
        and number of layers before max operation
        """
        with open('maxout.json', 'r') as f:
            self.hparams = json.load(f)
