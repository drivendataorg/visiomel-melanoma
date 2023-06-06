import contextlib
import copy
import functools
import math
import os
import os.path
import random
import warnings
from collections import OrderedDict
from typing import Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
import torchvision
import torchvision.models.resnet as resnet_factory
from torch.nn import (
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Conv3d,
    Dropout,
    Identity,
    InstanceNorm1d,
    LayerNorm,
    LeakyReLU,
    Linear,
    LogSoftmax,
    MaxPool2d,
    MaxPool3d,
    Module,
    ModuleList,
    MultiheadAttention,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
    functional,
)
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter
from torchvision import transforms
from torchvision.models import resnet18


def is_in_args(args, name, default):
    """Checks if the parammeter is specified in the args Namespace
    If not, attributes him the default value
    """
    if name in args.__dict__:
        para = getattr(args, name)
    else:
        para = default
    return para


class ResidualMLPBlock(Module):
    """
    Residual MLP block, with batch normalization and dropout.
    does not change size of input
    """

    def __init__(self, in_features, out_features=None, dropout=0, bn=True):
        super(ResidualMLPBlock, self).__init__()
        self.bn = bn
        if out_features is None:
            out_features = in_features
            self.residual_lin = Identity()
        else:
            self.residual_lin = LinearBatchNorm(
                in_features, out_features, dropout, constant_size=self.bn, relu=False
            )
        self.block = Sequential(
            LinearBatchNorm(in_features, out_features, dropout, constant_size=self.bn, relu=True),
            LinearBatchNorm(out_features, out_features, dropout, constant_size=self.bn, relu=False),
        )

    def forward(self, x):
        x = self.block(x) + self.residual_lin(x)
        x = ReLU()(x)
        return x


class ResidualMLP(Module):
    def __init__(self, args):
        super(ResidualMLP, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.feature_depth = is_in_args(args, "feature_depth", 256)
        self.num_class = is_in_args(args, "num_class", 2)
        self.n_layers_classif = 2  # is_in_args(args, 'n_layers_classif', 1)
        self.bn = is_in_args(args, "bn", True)
        tile_encoder = []
        tile_encoder.append(
            ResidualMLPBlock(self.feature_depth, 512, dropout=self.dropout, bn=self.bn)
        )
        for i in range(self.n_layers_classif - 1):
            tile_encoder.append(ResidualMLPBlock(512, dropout=self.dropout, bn=self.bn))
        tile_encoder = Sequential(*tile_encoder)
        self.tile_encoder = tile_encoder
        self.classifier = Linear(512, self.num_class)

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        if isinstance(x, tuple):
            x = x[0]
        bs, nbt, _ = x.shape
        x = self.tile_encoder(x)
        x = x.mean(1)
        out = self.classifier(x)
        return out


class LinearBatchNorm(Module):
    """
    Module defining a linear layer with batch normalization and dropout.

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        dropout (float): dropout rate
        constant_size (bool): if False, uses batch normalization, else uses instance normalization
    """

    def __init__(
        self, in_features, out_features, dropout, constant_size=True, dim_batch=None, relu=True
    ):
        if dim_batch is None:
            dim_batch = out_features
        super(LinearBatchNorm, self).__init__()
        self.cs = constant_size
        if relu:
            relu = ReLU()
        else:
            relu = Identity()
        self.block = Sequential(Linear(in_features, out_features), relu, Dropout(p=dropout))
        self.norm = self.get_norm(constant_size, dim_batch)

    def get_norm(self, constant_size, out_features):
        if not constant_size:
            print("Using instance norm")
            norm = Identity()  # InstanceNorm1d(out_features) --> InstanceNorm brings error in test
        else:
            norm = BatchNorm1d(out_features)
        return norm

    def forward(self, x):
        has_tile_dim = len(x.shape) > 2
        if has_tile_dim:
            bs, nbt, _ = x.shape
            x = x.view(bs * nbt, -1)
        x = self.block(x)
        x = self.norm(x)
        if has_tile_dim:
            x = x.view(bs, nbt, -1)
        return x
