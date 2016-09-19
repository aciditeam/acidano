#!/usr/bin/env python
# -*- coding: utf8 -*-

""" Theano CRBM implementation.

For details, see:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv
Sample data:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat

@author Graham Taylor"""

# Hyperopt
from hyperopt import hp
from math import log

# Numpy
import numpy as np
from numpy.random import RandomState

# Theano
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure


class cRBM(object):
    """Conditional Restricted Boltzmann Machine (CRBM)  """
    def __init__(self,
                 model_param,
                 dimensions,
                 weights_initialization=None):
        """ inspired by G. Taylor"""
        # Datas are represented like this:
        #   - visible : (num_batch, orchestra_dim)
        #   - past : (num_batch, orchestra_dim * (temporal_order-1) + piano_dim)
        self.batch_size = dimensions['batch_size']
        self.n_visible = dimensions['orchestra_dim']
        self.temporal_order = dimensions['temporal_order']
        self.n_past = (self.temporal_order-1) * dimensions['orchestra_dim'] + dimensions['piano_dim']

        # Number of hidden in the RBM
        self.n_hidden = model_param['n_hidden']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        self.rng_np = RandomState(25)

        # Weights
        if weights_initialization is None:
            self.W = shared_normal(self.n_visible, self.n_hidden, 0.01, self.rng_np)
            self.bv = shared_zeros(self.n_visible)
            self.bh = shared_zeros(self.n_hidden)
            self.A = shared_normal(self.n_past, self.n_visible, 0.01, self.rng_np)
            self.B = shared_normal(self.n_past, self.n_hidden, 0.01, self.rng_np)
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.A = weights_initialization['A']
            self.B = weights_initialization['B']

        self.params = [self.W, self.A, self.B, self.bv, self.bh]

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.v = T.fmatrix('v')
        self.p = T.fmatrix('p')

        self.rng = RandomStreams(seed=25)

        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################

    @staticmethod
    def get_hp_space():
        space = (hp.qloguniform('temporal_order', log(10), log(10), 1),
                 hp.qloguniform('n_hidden', log(100), log(5000), 10),
                 hp.quniform('batch_size', 100, 100, 1),
                 hp.qloguniform('gibbs_steps', log(1), log(50), 1),
                 )
        return space

    @staticmethod
    def get_param_dico(params):
        # Unpack
        if params is None:
            temporal_order, n_hidden, batch_size, gibbs_steps = [1,2,3,5]
        else:
            temporal_order, n_hidden, batch_size, gibbs_steps = params
        # Cast the params
        model_param = {
            'temporal_order': int(temporal_order),
            'n_hidden': int(n_hidden),
            'batch_size': int(batch_size),
            'gibbs_steps': int(gibbs_steps)
        }
        return model_param

    @staticmethod
    def name():
        return "cRBM"

    ###############################
    ##       NEGATIVE PARTICLE
    ###############################
    def free_energy(self, v, bv, bh):
        # sum along pitch axis
        fe = -(v * bv).sum(axis=1) - T.log(1 + T.exp(T.dot(v, self.W) + bh)).sum(axis=1)
        return fe

    def gibbs_step(self, v, bv, bh):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
        h = self.rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                              dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                              dtype=theano.config.floatX)
        return v, mean_v

    def get_negative_particle(self):
        # Get dynamic biases
        self.bv_dyn = T.dot(self.p, self.A) + self.bv
        self.bh_dyn = T.dot(self.p, self.B) + self.bh
        # Train the RBMs by blocks
        # Perform k-step gibbs sampling
        (v_chain, mean_chain), updates_rbm = theano.scan(
            fn=lambda v,bv,bh: self.gibbs_step(v, bv,bh),
            outputs_info=[self.v, None],
            non_sequences=[self.bv_dyn, self.bh_dyn],
            n_steps=self.k
        )
        # Get last element of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_chain[-1]

        return v_sample, mean_v, updates_rbm

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        # Get the negative particle
        v_sample, mean_v, updates_train = self.get_negative_particle()

        # Compute the free-energy for positive and negative particles
        fe_positive = self.free_energy(self.v, self.bv_dyn, self.bh_dyn)
        fe_negative = self.free_energy(v_sample, self.bv_dyn, self.bh_dyn)

        # Cost = mean along batches of free energy difference
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Monitor
        monitor = T.xlogx.xlogy0(self.v, mean_v) + T.xlogx.xlogy0(1 - self.v, 1 - mean_v)
        monitor = monitor.sum() / self.batch_size

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
        # updates_train = optimizer.get_updates(self.params, grads, updates_train)
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def build_past(self, piano, orchestra, index):
        # [T-1, T-2, ..., 0]
        decreasing_time = theano.shared(np.arange(self.temporal_order-1,0,-1, dtype=np.int32))
        #
        temporal_shift = T.tile(decreasing_time, (self.batch_size,1))
        # Reshape
        index_full = index.reshape((self.batch_size, 1)) - temporal_shift
        # Slicing
        past_orchestra = orchestra[index_full,:]\
            .ravel()\
            .reshape((self.batch_size, (self.temporal_order-1)*self.n_visible))
        present_piano = piano[index,:]
        # Concatenate along pitch dimension
        past = T.concatenate((present_piano, past_orchestra), axis=1)
        # Reshape
        return past

    def build_visible(self, orchestra, index):
        visible = orchestra[index,:]
        return visible

    def get_train_function(self, index, piano, orchestra, optimizer, name):
        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: self.build_visible(orchestra, index),
                                       self.p: self.build_past(piano, orchestra, index)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        # Generate the last frame for the sequence v
        v_sample, _, updates_valid = self.get_negative_particle()
        predicted_frame = v_sample
        # Get the ground truth
        true_frame = self.v
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)
        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    def get_validation_error(self, index, piano, orchestra, name):
        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v: self.build_visible(orchestra, index),
                                       self.p: self.build_past(piano, orchestra, index)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    #   Need no seed in this model
    ###############################
    #  def generate(self, k=20):
    #      # Random initialization of the visible units
    #      input_init = self.theano_rng.binomial(size=self.input.shape,
    #                                            n=1,
    #                                            p=0.5,
    #                                            dtype=theano.config.floatX)
    #      # compute positive phase
    #      pre_sigmoid_ph, ph_mean, ph_sample = \
    #          self.sample_h_given_v(input_init, self.input_history)
    #
    #      # for CD, we use the newly generate hidden sample
    #      chain_start = ph_sample
    #
    #      [pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means,
    #       nh_samples], updates = theano.scan(self.gibbs_hvh,
    #                                          outputs_info=[None, None, None, None, None, chain_start],
    #                                          non_sequences=self.input_history,
    #                                          n_steps=k)
    #
    #      mean_pred_v = nv_means[-1]
    #
    #      return mean_pred_v, updates
