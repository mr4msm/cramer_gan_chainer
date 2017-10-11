# -*- coding: utf-8 -*-
"""Critic, Generator, optimizers and hyper params definitions."""

import numpy as np
from chainer import Chain, Variable, cuda, initializers, optimizers
from chainer import functions as F
from chainer import links as L

Z_VECTOR_DIM = 128
BASE_CH = 64

HEIGHT = 64
WIDTH = 64
CHANNEL = 3

FLIP_H = True

BATCH_SIZE = 64
UPDATE_MAX = 40000
UPDATE_SAVE_PARAMS = 1000
UPDATE_CRI_PER_GEN = 1

LAMBDA = 10

OPTIMIZER_GEN = optimizers.Adam(alpha=1e-4, beta1=0., beta2=0.9)
OPTIMIZER_CRI = optimizers.Adam(alpha=1e-4, beta1=0., beta2=0.9)


class Generator(Chain):
    """Generator definition."""

    def __init__(self):
        """Prepare learnable layers."""
        super(Generator, self).__init__()
        with self.init_scope():
            self.fc0 = L.Linear(
                Z_VECTOR_DIM, 8 * BASE_CH * 4 * 4, nobias=False)
            self.deconv1 = L.Deconvolution2D(
                8 * BASE_CH, 4 * BASE_CH, 5, stride=2, pad=2, nobias=True)
            self.deconv2 = L.Deconvolution2D(
                4 * BASE_CH, 2 * BASE_CH, 5, stride=2, pad=2, nobias=True)
            self.deconv3 = L.Deconvolution2D(
                2 * BASE_CH, BASE_CH, 5, stride=2, pad=2, nobias=True)
            self.deconv4 = L.Deconvolution2D(
                BASE_CH, 3, 5, stride=2, pad=2, nobias=False)
            self.bn0 = L.BatchNormalization(8 * BASE_CH)
            self.bn1 = L.BatchNormalization(4 * BASE_CH)
            self.bn2 = L.BatchNormalization(2 * BASE_CH)
            self.bn3 = L.BatchNormalization(BASE_CH)

        for param in self.params():
            if param.name == 'W':
                p_norm = np.sqrt(np.sum(np.square(param.data)))
                param.data /= p_norm

    def __call__(self, z):
        """Neural net architecture, Define-by-RUN!"""
        x = F.reshape(self.fc0(z), (-1, 8 * BASE_CH, 4, 4))
        x = F.relu(self.bn0(x))
        height, width = x.data.shape[2:]

        height *= 2
        width *= 2
        self.deconv1.outsize = (height, width)
        x = F.relu(self.bn1(self.deconv1(x)))

        height *= 2
        width *= 2
        self.deconv2.outsize = (height, width)
        x = F.relu(self.bn2(self.deconv2(x)))

        height *= 2
        width *= 2
        self.deconv3.outsize = (height, width)
        x = F.relu(self.bn3(self.deconv3(x)))

        height *= 2
        width *= 2
        self.deconv4.outsize = (height, width)
        x = F.tanh(self.deconv4(x))

        return x


class Critic(Chain):
    """Critic definition."""

    def __init__(self):
        """Prepare learnable layers."""
        super(Critic, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(
                3, BASE_CH, 5, stride=2, pad=2, nobias=False)
            self.conv1 = L.Convolution2D(
                BASE_CH, 2 * BASE_CH, 5, stride=2, pad=2, nobias=True)
            self.conv2 = L.Convolution2D(
                2 * BASE_CH, 4 * BASE_CH, 5, stride=2, pad=2, nobias=True)
            self.conv3 = L.Convolution2D(
                4 * BASE_CH, 8 * BASE_CH, 5, stride=2, pad=2, nobias=True)
            self.fc4 = L.Linear(8 * BASE_CH * 4 * 4, 256, nobias=True)

        for param in self.params():
            if param.name == 'W':
                p_norm = np.sqrt(np.sum(np.square(param.data)))
                param.data /= p_norm

    def __call__(self, x):
        """Neural net architecture, Define-by-RUN!"""
        y = F.leaky_relu(self.conv0(x))
        y = F.leaky_relu(self.conv1(y))
        y = F.leaky_relu(self.conv2(y))
        y = F.leaky_relu(self.conv3(y))
        y = F.reshape(y, (-1, 8 * BASE_CH * 4 * 4))
        y = self.fc4(y)

        return y
