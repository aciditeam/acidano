#!/usr/bin/env python
# -*- coding: utf8 -*-

""" Theano CRBM implementation.

For details, see:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv
Sample data:
http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat

@author Graham Taylor"""

# Model lop
from model_lop import Model_lop

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


class cRBM(Model_lop):
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
            self.W = shared_normal(self.n_visible, self.n_hidden, 0.01, self.rng_np, name='W')
            self.bv = shared_zeros((self.n_visible), name='bv')
            self.bh = shared_zeros((self.n_hidden), name='bh')
            self.A = shared_normal(self.n_past, self.n_visible, 0.01, self.rng_np, name='A')
            self.B = shared_normal(self.n_past, self.n_hidden, 0.01, self.rng_np, name='B')
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.A = weights_initialization['A']
            self.B = weights_initialization['B']

        self.params = [self.W, self.A, self.B, self.bv, self.bh]

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.v = T.matrix('v', dtype=theano.config.floatX)
        self.p = T.matrix('p', dtype=theano.config.floatX)

        # v_gen : random init
        # p_gen : piano[t] ^ orchestra[t-N:t-1]
        self.v_gen = T.vector('v_gen', dtype=theano.config.floatX)
        self.p_gen = T.vector('p_gen', dtype=theano.config.floatX)
        self.v_gen.tag.test_value = np.random.rand(self.n_visible).astype(theano.config.floatX)
        self.p_gen.tag.test_value = np.random.rand(self.n_past).astype(theano.config.floatX)

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

    def get_negative_particle(self, v, p):
        # Get dynamic biases
        bv_dyn = T.dot(p, self.A) + self.bv
        bh_dyn = T.dot(p, self.B) + self.bh
        # Train the RBMs by blocks
        # Perform k-step gibbs sampling
        (v_chain, mean_chain), updates_rbm = theano.scan(
            fn=lambda v,bv,bh: self.gibbs_step(v, bv,bh),
            outputs_info=[v, None],
            non_sequences=[bv_dyn, bh_dyn],
            n_steps=self.k
        )
        # Get last element of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_chain[-1]

        return v_sample, mean_v, bv_dyn, bh_dyn, updates_rbm

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        # Get the negative particle
        v_sample, mean_v, bv_dyn, bh_dyn, updates_train = self.get_negative_particle(self.v, self.p)

        # Compute the free-energy for positive and negative particles
        fe_positive = self.free_energy(self.v, bv_dyn, bh_dyn)
        fe_negative = self.free_energy(v_sample, bv_dyn, bh_dyn)

        # Cost = mean along batches of free energy difference
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Monitor
        monitor = T.xlogx.xlogy0(self.v, mean_v) + T.xlogx.xlogy0(1 - self.v, 1 - mean_v)
        monitor = monitor.sum() / self.batch_size

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
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
        v_sample, _, _, _, updates_valid = self.get_negative_particle(self.v, self.p)
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

        # This time self.v is initialized randomly
        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v: (np.random.uniform(0, 1, (self.batch_size, self.n_visible))).astype(theano.config.floatX),
                                       self.p: self.build_past(piano, orchestra, index)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    ###############################
    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size,
                              name="generate_sequence"):
        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order

        # Graph for the negative particle
        v_sample, _, _, _, updates_nex_sample = \
            self.get_negative_particle(self.v_gen, self.p_gen)

        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.v_gen, self.p_gen],
            outputs=[v_sample],
            updates=updates_nex_sample,
            name="next_sample",
        )

        def closure(ind):
            start_piano_ind = ind - generation_length + seed_size
            orchestra_gen = np.zeros((generation_length, self.n_visible)).astype(theano.config.floatX)
            # Seed orchestra
            end_orch_init_ind = start_piano_ind
            start_orch_init_ind = end_orch_init_ind-(self.temporal_order-1)
            orchestra_gen[:self.temporal_order-1,:] = orchestra.get_value()[start_orch_init_ind:end_orch_init_ind,:]
            for index in xrange(self.temporal_order-1, generation_length, 1):
                index_piano = (ind - generation_length + 1) + index
                # Build past vector
                index_orchestra = np.arange(index-self.temporal_order+1, index, 1, dtype=np.int32)
                past_orchestra = orchestra_gen[index_orchestra,:].ravel()
                present_piano = piano.get_value()[index_piano,:]
                past_gen = np.concatenate((present_piano, past_orchestra))

                # Build initialisation vector
                visible_gen = (np.random.uniform(0, 1, self.n_visible)).astype(theano.config.floatX)

                # Get the next sample
                v_t = next_sample(visible_gen, past_gen)

                # Add this visible sample to the generated orchestra
                orchestra_gen[index,:] = np.asarray(v_t).astype(theano.config.floatX)

            return (orchestra_gen,)

        return closure
