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
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
        self.temporal_order = dimensions['temporal_order']
        self.n_piano = dimensions['piano_dim']
        self.n_orchestra = dimensions['orchestra_dim']
        #
        self.n_v = self.n_orchestra
        self.n_p = (self.temporal_order-1) * self.n_orchestra + self.n_piano
        self.n_h = model_param['n_h']

        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Regularization
        self.dropout_probability = model_param['dropout_probability']

        self.rng_np = RandomState(25)

        # Weights
        if weights_initialization is None:
            self.W = shared_normal((self.n_v, self.n_h), 0.01, self.rng_np, name='W')
            self.bv = shared_zeros((self.n_v), name='bv')
            self.bh = shared_zeros((self.n_h), name='bh')
            self.A = shared_normal((self.n_p, self.n_v), 0.01, self.rng_np, name='A')
            self.B = shared_normal((self.n_p, self.n_h), 0.01, self.rng_np, name='B')
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
        self.v_truth = T.matrix('v_truth', dtype=theano.config.floatX)

        self.v.tag.test_value = np.random.rand(self.batch_size, self.n_v).astype(theano.config.floatX)
        self.p.tag.test_value = np.random.rand(self.batch_size, self.n_p).astype(theano.config.floatX)

        # v_gen : random init
        # p_gen : piano[t] ^ orchestra[t-N:t-1]
        self.v_gen = T.matrix('v_gen', dtype=theano.config.floatX)
        self.p_gen = T.matrix('p_gen', dtype=theano.config.floatX)

        self.rng = RandomStreams(seed=25)

        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################

    @staticmethod
    def get_hp_space():
        space = (hp.qloguniform('temporal_order', log(10), log(10), 1),
                 hp.qloguniform('n_h', log(100), log(5000), 10),
                 hp.quniform('batch_size', 100, 100, 1),
                 hp.qloguniform('gibbs_steps', log(1), log(50), 1),
                 hp.choice('dropout', [
                     0.0,
                     hp.normal('dropout_probability', 0.5, 0.1)
                 ])
                 )
        return space

    @staticmethod
    def get_param_dico(params):
        # Unpack
        if params is None:
            temporal_order, n_h, batch_size, gibbs_steps, dropout_probability = [1,2,3,5,0.6]
        else:
            temporal_order, n_h, batch_size, gibbs_steps, dropout_probability = params
        # Cast the params
        model_param = {
            'temporal_order': int(temporal_order),
            'n_h': int(n_h),
            'batch_size': int(batch_size),
            'gibbs_steps': int(gibbs_steps),
            'dropout_probability': dropout_probability
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

    def gibbs_step(self, v, bv, bh, dropout_mask):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
        # Dropout
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h,
                              dtype=theano.config.floatX)

        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                              dtype=theano.config.floatX)
        return v, mean_v

    def get_negative_particle(self, v, p):
        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        else:
            dropout_mask = (1-self.dropout_probability)
        # Get dynamic biases
        bv_dyn = T.dot(p, self.A) + self.bv
        bh_dyn = T.dot(p, self.B) + self.bh
        # Train the RBMs by blocks
        # Perform k-step gibbs sampling
        (v_chain, mean_chain), updates_rbm = theano.scan(
            fn=self.gibbs_step,
            outputs_info=[v, None],
            non_sequences=[bv_dyn, bh_dyn, dropout_mask],
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
        monitor = T.nnet.binary_crossentropy(mean_v, self.v)
        monitor = monitor.sum() / self.batch_size

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def get_index_full(self, index, batch_size, length_seq):
        index.tag.test_value = np.linspace(50, 160, batch_size).astype(np.int32)
        # [T-1, T-2, ..., 0]
        decreasing_time = theano.shared(np.arange(length_seq-1,0,-1, dtype=np.int32))
        #
        temporal_shift = T.tile(decreasing_time, (batch_size,1))
        # Reshape
        index_full = index.reshape((batch_size, 1)) - temporal_shift
        return index_full

    def build_past(self, piano, orchestra, index, batch_size, length_seq):
        index_full = self.get_index_full(index, batch_size, length_seq)
        # Slicing
        past_orchestra = orchestra[index_full,:]\
            .ravel()\
            .reshape((batch_size, (length_seq-1)*self.n_v))
        present_piano = piano[index,:]
        # Concatenate along pitch dimension
        past = T.concatenate((present_piano, past_orchestra), axis=1)
        # Reshape
        return past

    def build_visible(self, orchestra, index):
        visible = orchestra[index,:]
        return visible

    def get_train_function(self, piano, orchestra, optimizer, name):

        super(cRBM, self).get_train_function()

        # index to a [mini]batch : int32
        index = T.ivector()

        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: self.build_visible(orchestra, index),
                                       self.p: self.build_past(piano, orchestra, index, self.batch_size, self.temporal_order)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v = self.rng.uniform((self.batch_size, self.n_v), 0, 1, dtype=theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, _, _, updates_valid = self.get_negative_particle(self.v, self.p)
        predicted_frame = v_sample
        # Get the ground truth
        true_frame = self.v_truth
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)
        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    def get_validation_error(self, piano, orchestra, name):

        super(cRBM, self).get_validation_error()

        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        # This time self.v is initialized randomly
        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v_truth: self.build_visible(orchestra, index),
                                       self.p: self.build_past(piano, orchestra, index, self.batch_size, self.temporal_order)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    ###############################
    def build_past_generation(self, piano_gen, orchestra_gen, index, batch_size, length_seq):
        past_orchestra = orchestra_gen[:,index-self.temporal_order+1:index,:]\
            .ravel()\
            .reshape((batch_size, (length_seq-1)*self.n_v))

        present_piano = piano_gen[:,index,:]
        p_gen = np.concatenate((present_piano, past_orchestra), axis=1)
        return p_gen

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):

        super(cRBM, self).get_generate_function()

        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order - 1

        self.v_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_v).astype(theano.config.floatX)
        self.p_gen.tag.test_value = np.random.rand(batch_generation_size, self.n_p).astype(theano.config.floatX)

        # Graph for the negative particle
        v_sample, _, _, _, updates_next_sample = \
            self.get_negative_particle(self.v_gen, self.p_gen)

        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.v_gen, self.p_gen],
            outputs=[v_sample],
            updates=updates_next_sample,
            name="next_sample",
        )

        def closure(ind):
            # Initialize generation matrice
            piano_gen, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for index in xrange(seed_size, generation_length, 1):
                # Build past vector
                p_gen = self.build_past_generation(piano_gen, orchestra_gen, index, batch_generation_size, self.temporal_order)
                # Build initialisation vector
                v_gen = (np.random.uniform(0, 1, (batch_generation_size, self.n_v))).astype(theano.config.floatX)
                # Get the next sample
                v_t = (np.asarray(next_sample(v_gen, p_gen))[0]).astype(theano.config.floatX)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:,index,:] = v_t

            return (orchestra_gen,)

        return closure
