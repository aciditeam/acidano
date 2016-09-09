#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Own code for the RnnRbm of Boulanger-Lewandowski
# Note that generation does not need a seed, but for validation we use
# a seed of size temporal_order and compare only the last generated sample
#  -> predictive performance

# Theano
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
# Hyperopt
from hyperopt import hp
from math import log
# Performance measures
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure
from acidano.utils.init import shared_normal, shared_zeros

import numpy as np

# from theano.printing import pydotprint

# Reproduce results
from numpy.random import RandomState


class RnnRbm(object):
    def __init__(self,
                 model_param,
                 dimensions,
                 weights_initialization=None):

        self.n_visible = dimensions['input_dim']
        self.batch_size = dimensions['batch_size']
        self.temporal_order = dimensions['temporal_order']
        # Number of hidden in the RBM
        self.n_hidden = model_param['n_hidden']
        # Number of hidden in the recurrent net
        self.n_hidden_recurrent = model_param['n_hidden_recurrent']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        self.rng_np = RandomState(25)

        # Weights
        if weights_initialization is None:
            self.W = shared_normal(self.n_visible, self.n_hidden, 0.01, self.rng_np)
            self.bv = shared_zeros(self.n_visible)
            self.bh = shared_zeros(self.n_hidden)
            self.Wuh = shared_normal(self.n_hidden_recurrent, self.n_hidden, 0.0001, self.rng_np)
            self.Wuv = shared_normal(self.n_hidden_recurrent, self.n_visible, 0.0001, self.rng_np)
            self.Wvu = shared_normal(self.n_visible, self.n_hidden_recurrent, 0.0001, self.rng_np)
            self.Wuu = shared_normal(self.n_hidden_recurrent, self.n_hidden_recurrent, 0.0001, self.rng_np)
            self.bu = shared_zeros(self.n_hidden_recurrent)
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.Wuh = weights_initialization['Wuh']
            self.Wuv = weights_initialization['Wuv']
            self.Wvu = weights_initialization['Wvu']
            self.Wuu = weights_initialization['Wuu']
            self.bu = weights_initialization['bu']

        self.params = [self.W, self.bv,self.bh, self.Wuh,
                       self.Wuv, self.Wvu, self.Wuu, self.bu]

        # Instanciate variables : (batch, time, pitch)
        # Note : we need the init variable to compile the theano function (get_train_function)
        # Indeed, self.v will be modified in the function, hence, giving a value to
        # self.v after these modifications does not set the value of the entrance node,
        # but set the value of the modified node
        self.v_init = T.tensor3('v')
        self.v = self.v_init
        self.v.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_visible).astype(theano.config.floatX)

        # Random generator
            # self.rng = RandomStreams(seed=np.random.randint(1 << 30))
        self.rng = RandomStreams(seed=25)
        self.u0 = T.zeros((self.batch_size, self.n_hidden_recurrent))  # initial value for the RNN hidden
        self.u0.tag.test_value = np.zeros((self.batch_size, self.n_hidden_recurrent), dtype=theano.config.floatX)

        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################

    @staticmethod
    def get_hp_space():
        space = (hp.qloguniform('temporal_order', log(100), log(100), 1),
                 hp.qloguniform('n_hidden', log(100), log(5000), 10),
                 hp.qloguniform('n_hidden_recurrent', log(100), log(5000), 10),
                 hp.quniform('batch_size', 100, 100, 1),
                 hp.qloguniform('gibbs_steps', log(1), log(50), 1),

                 #  hp.choice('activation_func', ['tanh', 'sigmoid']),
                 #  hp.choice('sampling_positive', ['true', 'false'])
                 # gibbs_sampling_step_test ???
                 )
        return space

    @staticmethod
    def get_param_dico(params):
        # Unpack
        if params is None:
            temporal_order, n_hidden, n_hidden_recurrent, batch_size, gibbs_steps = [1,2,3,4,5]
        else:
            temporal_order, n_hidden, n_hidden_recurrent, batch_size, gibbs_steps = params

        # Cast the params
        model_param = {
            'temporal_order': int(temporal_order),
            'n_hidden': int(n_hidden),
            'n_hidden_recurrent': int(n_hidden_recurrent),
            'batch_size': int(batch_size),
            'gibbs_steps': int(gibbs_steps)
        }

        return model_param

    @staticmethod
    def name():
        return "RnnRbm"

    ###############################
    ##       INFERENCE
    ###############################
    def free_energy(self, v, bv, bh):
        # sum along pitch and time axis
        fe = -(v * bv).sum(axis=(1,2)) - T.log(1 + T.exp(T.dot(v, self.W) + bh)).sum(axis=(1,2))
        # mean along time axis
        fe /= self.temporal_order
        return fe

    # def gibbs_step(self, v, bv, bh):
    def gibbs_step(self, v, bv,bh):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
        h = self.rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                              dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                              dtype=theano.config.floatX)
        return mean_v, v

    # Given v_t, and u_tm1 we can infer u_t
    def recurrence(self, v_t, u_tm1):
        bv_t = self.bv + T.dot(u_tm1, self.Wuv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh)
        u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))
        return [u_t, bv_t, bh_t]

    def inference(self):
        # A bit hacky
        # Re-initialize the visible unit (avoid copying useless dimshuffle
        # part of the graph computation of v)
        self.v = self.v_init
        # We have to dimshuffle so that time is the first dimension
        self.v = self.v.dimshuffle((1,0,2))

        # Write the recurrence to get the bias for the RBM
        (_, bv_t, bh_t), updates_inference = theano.scan(
            fn=self.recurrence,
            sequences=self.v, outputs_info=[self.u0, None, None])

        # Reshuffle the variables
        self.bv_dynamic = bv_t.dimshuffle((1,0,2))
        self.bh_dynamic = bh_t.dimshuffle((1,0,2))
        self.v = self.v.dimshuffle((1,0,2))

        # Train the RBMs by blocks
        # Perform k-step gibbs sampling
        v_chain, updates_rbm = theano.scan(
            fn=lambda v,bv,bh: self.gibbs_step(v,bv,bh)[1],
            outputs_info=[self.v],
            non_sequences=[self.bv_dynamic, self.bh_dynamic],
            n_steps=self.k
        )

        # Add updates of the rbm
        updates_inference.update(updates_rbm)

        # Get last sample of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = self.gibbs_step(v_sample,self.bv_dynamic,self.bh_dynamic)[0]

        return v_sample, mean_v, updates_inference

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        v_sample, mean_v, updates_train = self.inference()
        # Monitor function (binary cross-entropy)
        monitor = T.xlogx.xlogy0(self.v, mean_v) + T.xlogx.xlogy0(1 - self.v, 1 - mean_v)
        # monitor = T.binary_crossentropy(self.v, mean_v)

        # Compute cost function
        fe_positive = self.free_energy(self.v, self.bv_dynamic, self.bh_dynamic)
        fe_negative = self.free_energy(v_sample, self.bv_dynamic, self.bh_dynamic)

        # Mean along batches
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
        # updates_train = optimizer.get_updates(self.params, grads, updates_train)
        updates_train.update(
            ((p, p - 0.001 * g) for p, g in zip(self.params, grads))
        )

        return cost, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def build_train_matrix(self, train_pr, index):
        index.tag.test_value = np.linspace(50, 500, self.batch_size).astype(np.int32)
        # [T-1, T-2, ..., 0]
        decreasing_time = theano.shared(np.arange(self.temporal_order-1,-1,-1, dtype=np.int32))
        # Temporal_shift =
        #
        #        [i0-T+1   ; i1-T+1; i2-T+1 ; ... ; iN-T+1;
        #         i0-T+2 ;                  ; iN-T+2;
        #                       ...
        #         i0 ;                      ; iN]
        #
        #   with T = temporal_order
        #        N = pitch_order
        #
        temporal_shift = T.tile(decreasing_time, (self.batch_size,1))
        # Reshape
        index_full = index.reshape((self.batch_size, 1)) - temporal_shift
        # Slicing
        pr = train_pr[index_full.ravel(),:]
        # Reshape
        return T.reshape(pr, (self.batch_size, self.temporal_order, self.n_visible))

    def get_train_function(self, index, train_pr, optimizer, name):
        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost],
                               updates=updates,
                               givens={self.v_init: self.build_train_matrix(train_pr, index)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        # Generate the last frame for the sequence v
        v_sample, _, updates_valid = self.inference()
        predicted_frame = v_sample[:,-1,:]
        # Get the ground truth
        true_frame = self.v[:,-1,:]
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)

        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ###############################
    def get_validation_error(self, index, valid_pr, name):
        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v_init: self.build_train_matrix(valid_pr, index)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    #   Need no seed in this model
    ###############################
    # def recurrence_generation(u_tm1):
    #     bv_t = self.bv + T.dot(u_tm1, self.Wuv)
    #     bh_t = self.bh + T.dot(u_tm1, self.Wuh)
    #     u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))
    #     return [u_t, bv_t, bh_t]
    #
    # def inference_generation()
    # def generation(self):
