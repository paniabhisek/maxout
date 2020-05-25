#/usr/bin/python3
# -*- coding: utf-8 -*-

# Library modules
import json

# External library modules
import torch

# Local Library modules
from utils import *

class Maxout(torch.nn.Module):
    """Maxout using Multilayer Perceptron"""

    def __init__(self, input_size,
                 linear_layers, linear_neurons):
        """
        Define layers of maxout unit

        :param input_size: number of values(pixels or hidden unit's output)
                           that will be inputted to the layer
        :type input_size: :py:obj:`int`
        :param linear_layers: number of linear layers before
                              max operation
        :type linear_layers: :py:obj:`int`
        :param linear_neurons: number of neurons in each linear
                               layer before max operation
        :type linear_neurons: :py:obj:`int`
        """
        super(Maxout, self).__init__()

        # initialize variables
        self.input_size = input_size
        self.linear_layers = linear_layers
        self.linear_neurons = linear_neurons

        # batch normalization layer
        self.BN = torch.nn.BatchNorm1d(self.linear_neurons)

        # pytorch not able to reach the parameters of
        # linear layer inside a list
        self.params = torch.nn.ParameterList()
        self.z = []
        for layer in range(self.linear_layers):
            self.z.append(torch.nn.Linear(self.input_size,
                                          self.linear_neurons))
            self.params.extend(list(self.z[layer].parameters()))

    def forward(self, input_, is_norm=False, **kargs):
        """
        Function to forward inputs to maxout layer

        :param input_: input to the maxout layer
        :type input_: :py:class:`torch.Tensor`
        :param is_norm: whether to perform normalization before max
                        operation
        :type is_norm: :py:obj:`bool`
        :param kargs: keyword arguments containing
                       1. maximum value to allow to the next layer after normalization
                       2. :py:func:`torch.empty` containing the value norm constraint. It's size is same
                       as or broadcastable to the weight where norm constraint is performed.
        :type kargs: :py:obj:`dict`
        """
        h = None
        for layer in range(self.linear_layers):
            z = self.z[layer](input_)
            # norm + norm constraint
            if is_norm:
                z = self.BN(z)
                z = torch.where(z <= kargs['norm_constraint'],
                                z, kargs['norm_upper'])
            if layer == 0:
                h = z
            else:
                h = torch.max(h, z)
        return h
