#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import tensorflow as tf
import ipdb
import logging

log = logging.getLogger('encoder')


class CNNEncoder(object):
    def __init__(self, embedding_size=128, vocab_size=24,
                 stride=1, filternum=32, kernelsize=128, inputsize=2000,
                 poolstride=32, poolsize=64, outputsize=1024,
                 pretrained_embedding=None):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.stride = stride
        self.filternum = filternum
        self.kernelsize = kernelsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.outputsize = outputsize
        self.inputsize = inputsize
        self.pretrained_embedding = pretrained_embedding
        self.outputs = None

    def init_variables(self):
        self.xs_ = tf.placeholder(
            shape=[None, self.inputsize],
            dtype=tf.int32,
            name='x_in'
        )
        log.info('Encoder input shape {}'.format(self.xs_.shape))
        mask = tf.concat([[0], tf.ones(self.vocab_size - 1)], axis=0)
        log.info('Mask shape {}'.format(mask.shape))
        # input activation variables
        if hasattr(tf, 'initializers'):
            initializer = tf.initializers.random_uniform
        else:
            initializer = tf.random_uniform_initializer()

        if self.pretrained_embedding is None:
            self.emb = tf.get_variable(
                'emb',
                [self.vocab_size, self.embedding_size],
                dtype=tf.float32,
                initializer=initializer
            )
        else:
            self.emb = tf.get_variable(
                'emb',
                [self.vocab_size, self.embedding_size],
                dtype=tf.float32,
                initializer=self.pretrained_embedding,
                trainable=False
            )

        self.emb = tf.reshape(mask, shape=[-1, 1]) * self.emb
        log.info('Encoder Embedding  shape {} '.format(self.emb))

        # cnn kernel takes in shape [size,  (input channels, output channels)]
        self.cnnkernel = tf.get_variable(
            'kernel',
            [self.kernelsize, self.embedding_size, self.filternum],
            dtype=tf.float32
        )
        log.info('CNN kernel shape {}'.format(self.cnnkernel))

    def build(self):
        self.init_variables()

        self.cnn_inputs = tf.nn.dropout(
            tf.nn.embedding_lookup(
                self.emb,
                self.xs_,
                name='cnn_in'),
            0.2
        )
        log.info('CNN inputs {}'.format(self.cnn_inputs.shape))
        self.cnnout = tf.nn.relu(
            tf.nn.conv1d(
                self.cnn_inputs,
                self.cnnkernel,
                1,
                'VALID',
                data_format='NHWC',
                name='cnn1')
        )
        log.info('CNN outputs {}'.format(self.cnnout.shape))

        self.maxpool = tf.layers.max_pooling1d(self.cnnout, self.poolsize,
                                               self.poolstride, name='maxpool1')

        log.info('shape_cnnout-{}'.format(str(self.maxpool.get_shape())))
        # self.maxpool = tf.reshape(self.maxpool, shape=[self.maxpool.get_shape()[0], -1])
        # self.maxpool = tf.layers.Flatten()(self.maxpool)
        self.maxpool = tf.contrib.layers.flatten(self.maxpool)

        log.info('shape_maxpool-{}'.format(str((self.maxpool.get_shape()))))

        self.fcweights = tf.get_variable('fc1', shape=[self.maxpool.get_shape()[1],
                                                       self.outputsize],
                                         dtype=tf.float32)
        self.fcbias = tf.get_variable('fcbias', shape=[self.outputsize])
        self.outputs = tf.nn.relu(tf.matmul(self.maxpool, self.fcweights) + self.fcbias, name='enc_out')
        log.info('shape_encoderout-{}'.format(str((self.outputs.get_shape()))))
        return self
