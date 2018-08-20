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
joint_space_dim = 256

class joint_inf_decoder:

    def __init__(
            self,
            inputlayer,
            x_shape,
            y_shape,
            num_labels,
            n2v_emb_dim
    ):
        self.x_inp = inputlayer
        self.x_shape = x_shape

        self.y_shape = y_shape
        self.num_labels = num_labels
        self.n2v_emb_dim = n2v_emb_dim

        global joint_space_dim
        self.joint_space_dim = joint_space_dim
        return

    def init_variables(self,trainable=True):
        with tf.name_scope('inf_w_b'):

            initial_val_w1 = tf.truncated_normal(
                shape = [
                    self.x_dim ,
                    self.joint_space_dim
                ],
                stddev=1,
                dtype=tf.float32
            )

            self.W1 = tf.Variable(
                initial_value= initial_val_w1,
                trainable=trainable,
                name = 'W1'
            )

            initial_val_w2 = tf.truncated_normal(
                shape=[
                    self.n2v_emb_dim,
                    self.joint_space_dim
                ],
                stddev=1,
                dtype=tf.float32
            )

            self.W2 = tf.Variable(
                initial_value=initial_val_w2,
                trainable=trainable,
                name='W2'
            )

            initial_val_b1 = tf.truncated_normal(
                shape=[
                    self.num_labels,
                    self.joint_space_dim
                ],
                stddev=1,
                dtype=tf.float32
            )

            self.b1 = tf.Variable(
                initial_value=initial_val_b1,
                trainable=trainable,
                name='b1'
            )

            initial_val_b2 = tf.truncated_normal(
                shape=[
                    self.joint_space_dim
                ],
                stddev=1,
                dtype=tf.float32
            )

            self.b2 = tf.Variable(
                initial_value=initial_val_b2,
                trainable=trainable,
                name='b2'
            )

        return

    def setup_inputs(self):

        self.x_inp = tf.contrib.layers.flatten(self.x_inp,'x_inf_inp')
        y_shape = [None]
        y_shape.extend( self.y_shape)

        self.y_inp = tf.placeholder(
            dtype=tf.float32,
            shape=y_shape,
            name='y_inf_inp'
        )
        return



# ------------------------- # 

    def build(self , pretrained_dir = None):
        
        self.setup_inputs()
        
        # self.x_inp = tf.layers.flatten(self.x_inp)
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
        trainable = True
        if pretrained_dir is None:
            trainable = False

        self.init_variables(trainable)
        self.xw = tf.einsum('ijk,kl->ijl', _x, self.W1)
        self.xw_b = tf.add(self.xw,self.b1)
        print('xw plus b', self.xw_b.shape)

        self.y_vals = tf.einsum('ijk,kl->ijl', self.y_inp, self.W2)
        self.y_vals = tf.add(self.y_vals, self.b2)

        try:
            self.xw_b = tf.nn.l2_normalize(self.xw_b,dim=-1)
            self.y_vals = tf.nn.l2_normalize(self.y_vals,dim=-1)
        except:
            self.xw_b = tf.nn.l2_normalize(self.xw_b, axis=-1)
            self.y_vals = tf.nn.l2_normalize(self.y_vals, axis=-1)

        print( 'self.y_vals shape',self.y_vals.shape)
        loss = tf.losses.cosine_distance(self.xw_b, self.y_vals, dim=-1, reduction=tf.losses.Reduction.NONE)
        self.cos_loss = tf.reduce_mean(loss, axis=1, name='cos_loss')
        self.batch_loss = tf.reduce_mean(self.cos_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.005, name='final_optimizer')
        self.train = self.optimizer.minimize(self.cos_loss)
        tf.summary.scalar('loss', self.batch_loss)

        # ------------------------------------- #
        # Prediction
        # ------------------------------------- #
        self.cos_sim_threshold = tf.placeholder(
            dtype=tf.float32,
            shape=(),
            name="cos_sim_threshold"
        )
        self.y_labels  = tf.placeholder(
            dtype=tf.float32,
            shape = [None, self.num_labels ],
            name = "y_labels"
        )

        # try:
        #     self.xw_b = tf.nn.l2_normalize(self.xw_b,dim=-1)
        #     self.y_vals = tf.nn.l2_normalize(self.y_vals,dim=-1)
        # except:
        #     self.xw_b = tf.nn.l2_normalize(self.xw_b, axis=-1)
        #     self.y_vals = tf.nn.l2_normalize(self.y_vals, axis=-1)
        #
        # cos_dist = tf.losses.cosine_distance(self.xw_b, self.y_vals, dim=-1, reduction=tf.losses.Reduction.NONE)

        self.pred_labels = tf.to_int32(loss <= self.cos_sim_threshold)
        self.pred_labels = tf.squeeze(self.pred_labels, name="predictions")


        b_recall = tf.metrics.recall(
            self.y_labels,
            self.pred_labels,
            name='recall'
        )

        b_prec = tf.metrics.precision(
            self.y_labels,
            self.pred_labels,
            name='prec'
        )
        self.b_recall = tf.reduce_mean(b_recall)
        self.b_prec = tf.reduce_mean(b_prec)
        self.b_f1 = (2*(self.b_recall*self.b_prec)/(self.b_recall + self.b_prec))

        tf.summary.scalar('Precision', self.b_prec)
        tf.summary.scalar('Recall', self.b_recall)
        tf.summary.scalar('F1', self.b_f1)
        self.summary = tf.summary.merge_all()
        self.names = [n.name + ':0' for n in tf.get_default_graph().as_graph_def().node]
        return

    def return_wt_list_to_restore(self):
        return self.names
