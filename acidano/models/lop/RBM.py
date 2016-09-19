#!/usr/bin/env python
# -*- coding: utf8 -*-

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


class RBM(object):
    """ A lop adapted RBM.
    Generation is performed by inpainting with some of the visible units clamped (context and piano)"""

    def __init__(self,
                 model_param,
                 dimensions,
                 weights_initialization=None):
        # Datas are represented like this:
        #   - visible = concatenation of the data : (num_batch, piano ^ orchestra_dim * temporal_order)
        self.batch_size = dimensions['batch_size']
        self.temporal_order = dimensions['temporal_order']
        self.n_v = dimensions['orchestra_dim']
        self.n_c = (self.temporal_order - 1) * dimensions['orchestra_dim'] + dimensions['piano_dim']

        # Number of hidden in the RBM
        self.n_h = model_param['n_hidden']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        self.rng_np = RandomState(25)

        # Weights
        if weights_initialization is None:
            self.W = shared_normal(self.n_v, self.n_hidden, 0.01, self.rng_np)
            self.C = shared_normal(self.n_c, self.n_hidden, 0.01, self.rng_np)
            self.bv = shared_zeros(self.n_v)
            self.bc = shared_zeros(self.n_c)
            self.bh = shared_zeros(self.n_h)
        else:
            self.W = weights_initialization['W']
            self.C = weights_initialization['C']
            self.bv = weights_initialization['bv']
            self.bc = weights_initialization['bc']
            self.bh = weights_initialization['bh']

        self.params = [self.W, self.C, self.bv, self.bc, self.bh]

        # We distinguish between clamped (= C as clamped or context) and non clamped units (= V as visible units)
        self.v = T.fmatrix('v')
        self.c = T.fmatrix('c')

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
            temporal_order, n_hidden, batch_size, gibbs_steps = [1,2,3,4]
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
        return "RBM (generation by inpainting)"

    ###############################
    ##       NEGATIVE PARTICLE
    ###############################
    def free_energy(self, v, c):
        # sum along pitch axis
        fe = -(v*self.bv + c*self.bc).sum(axis=1) - T.log(1 + T.exp(T.dot(v, self.W) + T.dot(c, self.C) + self.bh)).sum(axis=1)
        return fe

    def gibbs_step(self, v, c):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + T.dot(c, self.C) + self.bh)
        h = self.rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                              dtype=theano.config.floatX)
        v_mean = T.nnet.sigmoid(T.dot(h, self.W.T) + self.bv)
        v = self.rng.binomial(size=v_mean.shape, n=1, p=v_mean,
                              dtype=theano.config.floatX)
        c_mean = T.nnet.sigmoid(T.dot(h, self.C.T) + self.bc)
        c = self.rng.binomial(size=c_mean.shape, n=1, p=c_mean,
                              dtype=theano.config.floatX)
        return v, v_mean, c, c_mean

    def get_negative_particle(self):
        # Perform k-step gibbs sampling
        (v_chain, v_chain_mean, c_chain, c_chain_mean), updates_rbm = theano.scan(
            fn=lambda v,c: self.gibbs_step(v,c),
            outputs_info=[self.v, None, self.c, None],
            n_steps=self.k
        )
        # Get last element of the gibbs chain
        v_sample = v_chain[-1]
        c_sample = c_chain[-1]
        _, v_mean, _, c_mean = self.gibbs_step(v_sample, c_sample)

        return v_sample, v_mean, c_sample, c_mean, updates_rbm

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        # Get the negative particle
        v_sample, v_mean, c_sample, c_mean, updates_train = self.get_negative_particle()

        # Compute the free-energy for positive and negative particles
        fe_positive = self.free_energy(self.v, self.c)
        fe_negative = self.free_energy(v_sample, c_sample)

        # Cost = mean along batches of free energy difference
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Monitor
        monitor = T.xlogx.xlogy0(self.v, v_mean) + T.xlogx.xlogy0(1 - self.v, 1 - v_mean) +\
            T.xlogx.xlogy0(self.c, c_mean) + T.xlogx.xlogy0(1 - self.c, 1 - c_mean)
        monitor = monitor.sum() / self.batch_size

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample, c_sample])
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def build_context(self, piano, orchestra, index):
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
                                       self.c: self.build_context(piano, orchestra, index)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        # Generate the last frame for the sequence v
        v_sample, _, c_sample, _ updates_valid = self.get_negative_particle()
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
                                       self.c: self.build_context(piano, orchestra, index)},
                               name=name
                               )

    ###############################
    ##       GENERATION
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
