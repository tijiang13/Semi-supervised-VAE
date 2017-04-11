# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import logpdf
import time

from sklearn.metrics import accuracy_score

np.random.seed(271828)
random.seed(271828)
tf.set_random_seed(271828)

import tensorflow.contrib.layers as tcl
class VAE:
    def __init__(self):
        self.name = 'VAE'

        # classifier params
        self.hidden_size = 600

        # encode & decoder params
        self.z_dim = 50
        self.x_dim = 28*28

        # learning rate
        self.lr = 3e-4
        # first moment decay
        self.beta1 = 1-1e-1
        # second moment decay
        self.beta2 = 1-1e-3

        self._build_graph()
        self._build_train_op()
        self.check_parameters()

    def check_parameters(self):
        for var in tf.trainable_variables():
            print('%s: %s' % (var.name, var.get_shape()))
        print()

    def get_collection(self, collections):
        return [var for var in tf.get_collection(collections)]

    def reparameterize(self, mu, logvar):
        batch_size, eps_dim = tf.shape(mu)[0], tf.shape(mu)[1]
        std = tf.exp(logvar * 0.5)
        eps = tf.random_normal([batch_size, eps_dim])
        z = mu + eps * std
        return z

    def encode(self, x, reuse = False):
        with tf.variable_scope(self.name+'/encoder', reuse = reuse):
            h1 = tcl.fully_connected(x,  self.hidden_size, activation_fn = tf.nn.softplus)
            h1 = tf.nn.dropout(h1, keep_prob = self.encode_keep_prob)
            h2 = tcl.fully_connected(h1, self.hidden_size, activation_fn = tf.nn.softplus)
            h2 = tf.nn.dropout(h2, keep_prob = self.encode_keep_prob)
            mu     = tcl.fully_connected(h2, self.z_dim, activation_fn = None)
            logvar = tcl.fully_connected(h2, self.z_dim, activation_fn = None)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar

    def decode(self, z, reuse = False):
        with tf.variable_scope(self.name+'/decoder', reuse = reuse):
            h1 = tcl.fully_connected(z,  self.hidden_size, activation_fn = tf.nn.softplus)
            h1 = tf.nn.dropout(h1, keep_prob = self.decode_keep_prob)
            h2 = tcl.fully_connected(h1, self.hidden_size, activation_fn = tf.nn.softplus)
            h2 = tf.nn.dropout(h2, keep_prob = self.decode_keep_prob)
            x  = tcl.fully_connected(h2, self.x_dim, activation_fn = tf.nn.softmax)
            return x

    def L(self, x, recon_x, z, mu_z, logvar_z):
        # (batch_size, z_dim) -> batch_size,
        kld = tf.reduce_sum(logpdf.KLD(mu_z, logvar_z), 1)
        # (batch_size, 784)   -> batch_size,
        logpx = tf.reduce_sum(logpdf.bernoulli(recon_x, x), 1)
        loss = kld - logpx
        return loss

    def _build_graph(self, reuse = False):
        self.x = tf.placeholder(tf.float32, shape = (None, self.x_dim))
        self.encode_keep_prob = tf.placeholder(tf.float32)
        self.decode_keep_prob = tf.placeholder(tf.float32)
        '''
            labelled data, encoder & decoder
        '''
        # encoder, labelled data
        self.z, self.mu_z, self.logvar_z = self.encode(self.x, reuse = reuse)

        # decoder, labelled data
        self.x_recon = self.decode(self.z, reuse = reuse)

        # loss of labelled data, refered as L(x, y)
        self.loss = self.L(self.x, self.x_recon, self.z, self.mu_z, self.logvar_z)
        self.loss = tf.reduce_mean(self.loss, 0)

        trainable_vars_key = tf.GraphKeys.TRAINABLE_VARIABLES
        encoder_vars = tf.get_collection(key=trainable_vars_key, scope=self.name+"/encoder")
        decoder_vars = tf.get_collection(key=trainable_vars_key, scope=self.name+"/decoder")
        tcl.apply_regularization(tcl.l2_regularizer(1.0), encoder_vars)
        tcl.apply_regularization(tcl.l2_regularizer(1.0), decoder_vars)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 1e-6
        self.loss += reg_constant * tf.reduce_sum(reg_losses)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3,\
            pad_step_number=True, keep_checkpoint_every_n_hours=5.0)

    def _build_train_op(self):
        self.global_step = tf.Variable(0, name="global_step", trainable = False)
        optimizer = tf.train.AdamOptimizer(self.lr, self.beta1, self.beta2)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)
        capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(capped_gvs, self.global_step)

    def _build_generator_op(self):
        self._z   = tf.placeholder(tf.float32, shape = (None, self.z_dim))
        self._gen = self.decode(self._z, reuse = True)

    def optimize(self, sess, x):
        feed_dict = {
            self.x: x,
            self.encode_keep_prob: 1,
            self.decode_keep_prob: 1,
        }
        _, loss = sess.run([self.train_op, self.loss], feed_dict = feed_dict)
        return loss

    def recon_x(self, sess, x):
        feed_dict = {
            self.x: x,
            self.encode_keep_prob: 1,
            self.decode_keep_prob: 1,
        }
        x_recon = sess.run([self.x_recon], feed_dict = feed_dict)[0]
        return x_recon

    def generate(self, sess, z):
        feed_dict = {
            self._z : z,
            self.decode_keep_prob: 1,
        }
        _gen = sess.run([self._gen], feed_dict = feed_dict)[0]
        return _gen

    def save(self, sess, path = 'models/vae/ckpt'):
        self.saver.save(sess, path, global_step = self.global_step)

    def get_repr(self, sess, x, stats_only = False):
        feed_dict = {
            self.x: x,
            self.encode_keep_prob: 1,
            self.decode_keep_prob: 1,
        }
        if stats_only:
            mu, logvar = sess.run([self.mu_z, self.logvar_z], feed_dict = feed_dict)
            return mu, logvar
        else:
            z = sess.run([self.z], feed_dict = feed_dict)[0]
            return z
