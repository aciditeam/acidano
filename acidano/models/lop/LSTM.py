#!/usr/bin/env python
# -*- coding: utf8 -*-

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

# Propagation
from acidano.utils.forward import propup_sigmoid, propup_tanh
# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure


class LSTM(Model_lop):
    """ LSTM for LOP
    Predictive model,
        input = piano(t)
        output = orchestra(t)
        measure = cross-entropy error function
            (output units are binary units (y_j) considered independent : i != j -> y_j indep y_i)
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 weights_initialization=None):
        # Datas are represented like this:
        #   - visible = concatenation of the data : (num_batch, piano ^ orchestra_dim * temporal_order)
        self.batch_size = dimensions['batch_size']
        self.temporal_order = dimensions['temporal_order']
        self.n_v = dimensions['piano_dim']
        self.n_o = dimensions['orchestra_dim']

        # Number of hidden in the RBM
        self.n_h = model_param['n_hidden']

        # Numpy and theano random generators
        self.rng_np = RandomState(25)
        self.rng = RandomStreams(seed=25)

        if weights_initialization is None:
            # Weights
            # input gate
            self.L_vi = shared_normal(self.n_v, self.n_h, 0.01, name='L_vi')
            self.L_hi = shared_normal(self.n_h, self.n_h, 0.01, name='L_hi')
            self.b_i = shared_zeros((self.n_h), name='b_i')
            # Internal cell
            self.L_vc = shared_normal(self.n_v, self.n_h, 0.01, name='L_vc')
            self.L_hc = shared_normal(self.n_h, self.n_h, 0.01, name='L_hc')
            self.b_c = shared_zeros((self.n_h), name='b_c')
            # Forget gate
            self.L_vf = shared_normal(self.n_v, self.n_h, 0.01, name='L_vf')
            self.L_hf = shared_normal(self.n_h, self.n_h, 0.01, name='L_hf')
            self.b_f = shared_zeros((self.n_h), name='b_f')
            # Output
            # No L_co... as in Theano tuto
            self.L_vo = shared_normal(self.n_v, self.n_h, 0.01, name='L_vo')
            self.L_ho = shared_normal(self.n_h, self.n_h, 0.01, name='L_ho')
            self.b_o = shared_zeros((self.n_h), name='b_o')

            # Last predictive layer
            self.W = shared_normal(self.n_h, self.n_o, 0.01, name='W')
            self.b = shared_zeros((self.n_o), name='b')
        else:
            self.L_vi = weights_initialization['L_vi']
            self.L_hi = weights_initialization['L_hi']
            self.b_i = weights_initialization['b_i']
            self.L_vc = weights_initialization['L_vc']
            self.L_hc = weights_initialization['L_hc']
            self.b_c = weights_initialization['b_c']
            self.L_vf = weights_initialization['L_vf']
            self.L_hf = weights_initialization['L_hf']
            self.b_f = weights_initialization['b_f']
            self.L_vo = weights_initialization['L_vo']
            self.L_ho = weights_initialization['L_ho']
            self.b_o = weights_initialization['b_o']
            self.W = weights_initialization['W']
            self.b = weights_initialization['b']

        self.params = [self.L_vi, self.L_hi, self.b_i, self.L_vc, self.L_hc,
                       self.b_c, self.L_vf, self.L_hf, self.b_f, self.L_vo,
                       self.L_ho, self.b_o, self.W, self.b]

        # Variables
        self.v = T.tensor3('v')
        self.o = T.tensor3('o')

        # Initialize hidden states
        self.h_0 = T.zeros((self.batch_size, self.n_h))
        self.c_0 = T.zeros((self.batch_size, self.n_h))

        self.v.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)
        self.o.tag.test_value = np.random.rand(self.batch_size, self.temporal_order, self.n_o).astype(theano.config.floatX)
        self.h_0.tag.test_value = np.zeros((self.batch_size, self.n_h)).astype(theano.config.floatX)
        self.c_0.tag.test_value = np.zeros((self.batch_size, self.n_h)).astype(theano.config.floatX)
        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():
        space = (hp.qloguniform('temporal_order', log(10), log(100), 1),
                 hp.qloguniform('n_hidden', log(100), log(5000), 10),
                 hp.quniform('batch_size', 100, 100, 1),
                 )
        return space

    @staticmethod
    def get_param_dico(params):
        # Unpack
        if params is None:
            temporal_order, n_hidden, batch_size = [1,2,3]
        else:
            temporal_order, n_hidden, batch_size = params
        # Cast the params
        model_param = {
            'temporal_order': int(temporal_order),
            'n_hidden': int(n_hidden),
            'batch_size': int(batch_size),
        }
        return model_param

    @staticmethod
    def name():
        return "LSTM__1_layer"

    ###############################
    ##  FORWARD PASS
    ###############################
    def iteration(self, v_t, c_tm1, h_tm1):
        # Input gate
        i = propup_sigmoid(T.concatenate([v_t, h_tm1], axis=1), T.concatenate([self.L_vi, self.L_hi]), self.b_i)
        # Forget gate
        f = propup_sigmoid(T.concatenate([v_t, h_tm1], axis=1), T.concatenate([self.L_vf, self.L_hf]), self.b_f)
        # Cell update term
        c_tilde = propup_tanh(T.concatenate([v_t, h_tm1], axis=1), T.concatenate([self.L_vc, self.L_hc]), self.b_c)
        c_t = f * c_tm1 + i * c_tilde
        # Output gate
        o = propup_sigmoid(T.concatenate([v_t, h_tm1], axis=1), T.concatenate([self.L_vo, self.L_ho]), self.b_o)
        # h_t
        h_t = o * T.tanh(c_t)
        return c_t, h_t

    def forward_pass(self):
        # Time needs to be the first dimension
        v_loop = self.v.dimshuffle((1,0,2))
        # Infer hidden states
        (c_seq, h_seq), updates = theano.scan(fn=self.iteration,
                                              sequences=[v_loop],
                                              outputs_info=[self.c_0, self.h_0])

        # (batch, time, pitch)
        h_seq = h_seq.dimshuffle((1,0,2))

        # Activation probability
        o_mean = propup_sigmoid(h_seq, self.W, self.b)
        # Sample
        o_sample = self.rng.binomial(size=o_mean.shape, n=1, p=o_mean,
                                     dtype=theano.config.floatX)

        return o_mean, o_sample, updates

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        # Infer Orchestra sequence
        self.pred, _, updates_train = self.forward_pass()

        # Compute error function
        cost = T.nnet.binary_crossentropy(self.pred, self.o)
        # Sum over time and pitch axis
        cost = cost.sum(axis=(1,2))
        # Mean along batch dimension
        cost = T.mean(cost)

        # Monitor = cost normalized by sequence length
        monitor = cost / self.temporal_order

        # Update weights
        grads = T.grad(cost, self.params)
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def build_sequence(self, pr, index, last_dim):
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
        pr = pr[index_full.ravel(),:]
        # Reshape
        return T.reshape(pr, (self.batch_size, self.temporal_order, last_dim))

    def get_train_function(self, index, piano, orchestra, optimizer, name):
        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v: self.build_sequence(piano, index, self.n_v),
                                       self.o: self.build_sequence(orchestra, index, self.n_o)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        # Generate the last frame for the sequence v
        _, predicted_frame, updates_valid = self.forward_pass()
        # Get the ground truth
        true_frame = self.o
        # Measure the performances
        precision_time = precision_measure(true_frame, predicted_frame)
        recall_time = recall_measure(true_frame, predicted_frame)
        accuracy_time = accuracy_measure(true_frame, predicted_frame)
        # 2 options :
        #       1 - take the last time index
        # precision = precision_time[:,-1]
        # recall = recall_time[:,-1]
        # accuracy = accuracy_time[:,-1]
        #       2 - mean over time
        precision = T.mean(precision_time, axis=1)
        recall = T.mean(recall_time, axis=1)
        accuracy = T.mean(accuracy_time, axis=1)
        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    def get_validation_error(self, index, piano, orchestra, name):
        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v: self.build_sequence(piano, index, self.n_v),
                                       self.o: self.build_sequence(orchestra, index, self.n_o)},
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
