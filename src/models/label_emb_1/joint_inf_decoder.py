"""
    *.py: Description of what * does.
    Last Modified:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "0.0.1"

import tensorflow as tf
import numpy as np
import pandas as pd


class joint_inf_decoder:

    def __init__(
            self,
            x_shape,
            y_shape,
            num_labels,
            w2v_emb_dim
    ):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.num_labels = num_labels
        self.w2v_emb_dim = w2v_emb_dim
        return

    def init_variables(self):


        with tf.name_scope('inf_w_b'):

            initial_val = tf.truncated_normal(
                shape = [
                    self.x_dim ,
                    self.w2v_emb_dim
                ],
                stddev=0.1,
                dtype=tf.float64
            )

            self.W = tf.Variable(
                initial_value= initial_val,
                trainable=True,
                name = 'W'
            )
        return

    def setup_inputs(self):
        x_shape = [None]
        x_shape.extend(self.x_shape)
        self.x_inp = tf.placeholder(
            dtype=tf.float64,
            shape=x_shape,
            name='x_inf_inp'
        )

        y_shape = [None]
        y_shape.extend( self.y_shape)
        self.y_inp = tf.placeholder(
            dtype=tf.float64,
            shape=y_shape,
            name='y_inf_inp'
        )

    def build(self):
        self.setup_inputs()
        self.x_inp = tf.layers.flatten(self.x_inp)
        self.x_dim = self.x_inp.shape.as_list()[-1]
        '''
        if x has dim [ ?  , 8, from cae.encoder_op 
        x = [? , 1024] -> x [ ?, num_lablels, 1024 ]
        '''
        _x = tf.reshape(
            tf.tile(
                self.x_inp,
                [1,self.num_labels]
            ),
            [-1,self.num_labels,self.x_dim]
        )
        self.init_variables()
        xw = tf.einsum('ijk,kl->ijl', _x, self.W)
        loss = tf.losses.cosine_distance(xw,self.y_inp,axis=-1,reduction=tf.losses.Reduction.NONE)

        self.loss = tf.reduce_sum(loss,axis=1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train = self.optimizer.minimize(self.loss)

        return
