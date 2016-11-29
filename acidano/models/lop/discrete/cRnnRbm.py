#!/usr/bin/env python
# -*- coding: utf8 -*-

# Model lop
from acidano.models.lop.model_lop import Model_lop

# Hyperopt
from hyperopt import hp
from math import log

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure


class cRnnRbm(Model_lop):
    """ RnnRbm for LOP
    Predictive model,
        visible = orchestra(t) ^ piano(t)
        cost = free-energy between positive and negative particles
    """

    def __init__(self,
                 model_param,
                 dimensions,
                 weights_initialization=None):

        super(cRnnRbm, self).__init__(model_param, dimensions)

        # Number of visible units
        self.n_v = dimensions['orchestra_dim']
        # Number of context units
        self.n_c = dimensions['piano_dim']
        # Number of hidden in the RBM
        self.n_h = model_param['n_hidden']
        # Number of hidden in the recurrent net
        self.n_u = model_param['n_hidden_recurrent']
        # Number of Gibbs sampling steps
        self.k = model_param['gibbs_steps']

        # Weights
        if weights_initialization is None:
            # RBM weights
            self.W = shared_normal((self.n_v, self.n_h), 0.01, self.rng_np, name='W')
            self.bv = shared_zeros(self.n_v, name='bv')
            self.bh = shared_zeros(self.n_h, name='bh')
            # Conditional weights
            self.Wcv = shared_normal((self.n_c, self.n_v), 0.01, self.rng_np, name='Wcv')
            self.Wch = shared_normal((self.n_c, self.n_h), 0.01, self.rng_np, name='Wch')
            # Temporal weights
            self.Wuh = shared_normal((self.n_u, self.n_h), 0.0001, self.rng_np, name='Wuh')
            self.Wuv = shared_normal((self.n_u, self.n_v), 0.0001, self.rng_np, name='Wuv')
            self.Wvu = shared_normal((self.n_v, self.n_u), 0.0001, self.rng_np, name='Wvu')
            self.Wuu = shared_normal((self.n_u, self.n_u), 0.0001, self.rng_np, name='Wuu')
            self.bu = shared_zeros(self.n_u, name='bu')
        else:
            self.W = weights_initialization['W']
            self.bv = weights_initialization['bv']
            self.bh = weights_initialization['bh']
            self.Wcv = weights_initialization['Wcv']
            self.Wch = weights_initialization['Wch']
            self.Wuh = weights_initialization['Wuh']
            self.Wuv = weights_initialization['Wuv']
            self.Wvu = weights_initialization['Wvu']
            self.Wuu = weights_initialization['Wuu']
            self.bu = weights_initialization['bu']

        self.params = [self.W, self.bv, self.bh, self.Wcv, self.Wch,
                       self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu]

        # Instanciate variables : (batch, time, pitch)
        # Note : we need the init variable to compile the theano function (get_train_function)
        # Indeed, self.v will be modified in the function, hence, giving a value to
        # self.v after these modifications does not set the value of the entrance node,
        # but set the value of the modified node
        self.v_init = T.tensor3('v', dtype=theano.config.floatX)
        self.v_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)
        self.c_init = T.tensor3('o', dtype=theano.config.floatX)
        self.c_init.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_c).astype(theano.config.floatX)
        self.v_truth = T.tensor3('o_truth', dtype=theano.config.floatX)
        self.v_truth.tag.test_value = self.rng_np.rand(self.batch_size, self.temporal_order, self.n_v).astype(theano.config.floatX)

        # Generation Variables
        self.v_seed = T.tensor3('v_seed', dtype=theano.config.floatX)
        self.c_seed = T.tensor3('c_seed', dtype=theano.config.floatX)
        self.u_gen = T.matrix('u_gen', dtype=theano.config.floatX)
        self.c_gen = T.matrix('c_gen', dtype=theano.config.floatX)

        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()

        space = super_space + (hp.qloguniform('n_hidden', log(100), log(5000), 10),
                               hp.qloguniform('n_hidden_recurrent', log(100), log(5000), 10),
                               hp.qloguniform('gibbs_steps', log(1), log(50), 1),
                               )
        return space

    @staticmethod
    def get_param_dico(params):
        # Unpack
        if params is None:
            batch_size, temporal_order, dropout_probability, weight_decay_coeff, n_hidden, n_hidden_recurrent, gibbs_steps = [1,2,0.1,0.2,3,4,5]
        else:
            batch_size, temporal_order, dropout_probability, weight_decay_coeff, n_hidden, n_hidden_recurrent, gibbs_steps = params

        # Cast the params
        model_param = {
            'temporal_order': int(temporal_order),
            'n_hidden': int(n_hidden),
            'dropout_probability': dropout_probability,
            'weight_decay_coeff': weight_decay_coeff,
            'n_hidden_recurrent': int(n_hidden_recurrent),
            'batch_size': int(batch_size),
            'gibbs_steps': int(gibbs_steps)
        }

        return model_param

    @staticmethod
    def name():
        return "cRnnRbm"

    ###############################
    ##       INFERENCE
    ###############################
    def free_energy(self, v, bv, bh):
        # sum along pitch axis (last axis)
        last_axis = v.ndim - 1
        A = -(v*bv).sum(axis=last_axis)
        C = -(T.log(1 + T.exp(T.dot(v, self.W) + bh))).sum(axis=last_axis)
        fe = A + C
        return fe

    def gibbs_step(self, v, bv, bh, dropout_mask):
        # bv and bh defines the dynamic biases computed thanks to u_tm1
        mean_h = T.nnet.sigmoid(T.dot(v, self.W) + bh)
        # Dropout
        mean_h_corrupted = T.switch(dropout_mask, mean_h, 0)
        h = self.rng.binomial(size=mean_h_corrupted.shape, n=1, p=mean_h_corrupted,
                              dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, self.W.T) + bv)
        v = self.rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                              dtype=theano.config.floatX)
        return mean_v, v

    # Given v_t, and u_tm1 we can infer u_t
    def recurrence(self, v_t, c_t, u_tm1):
        bv_t = self.bv + T.dot(u_tm1, self.Wuv) + T.dot(c_t, self.Wcv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh) + T.dot(c_t, self.Wch)
        u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))
        return [u_t, bv_t, bh_t]

    def rnn_inference(self, v_init, c_init, u0):
        # We have to dimshuffle so that time is the first dimension
        v = v_init.dimshuffle((1,0,2))
        c = c_init.dimshuffle((1,0,2))

        # Write the recurrence to get the bias for the RBM
        (u_t, bv_t, bh_t), updates_dynamic_biases = theano.scan(
            fn=self.recurrence,
            sequences=[v, c], outputs_info=[u0, None, None])

        # Reshuffle the variables and keep trace
        self.bv_dynamic = bv_t.dimshuffle((1,0,2))
        self.bh_dynamic = bh_t.dimshuffle((1,0,2))

        return u_t, updates_dynamic_biases

    def inference(self, v, c):
        # Infer the dynamic biases
        u0 = T.zeros((self.batch_size, self.n_u))  # initial value for the RNN hidden
        u0.tag.test_value = np.zeros((self.batch_size, self.n_u), dtype=theano.config.floatX)
        u_t, updates_rnn_inference = self.rnn_inference(v, c, u0)

        # Train the RBMs by blocks
        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        else:
            dropout_mask = (1-self.dropout_probability)
        # Perform k-step gibbs sampling
        (mean_v_chain, v_chain), updates_inference = theano.scan(
            fn=lambda v,bv,bh: self.gibbs_step(v, bv, bh, dropout_mask),
            outputs_info=[None, v],
            non_sequences=[self.bv_dynamic, self.bh_dynamic],
            n_steps=self.k
        )

        # Add updates of the rbm
        updates_inference.update(updates_rnn_inference)

        # Get last sample of the gibbs chain
        v_sample = v_chain[-1]
        mean_v = mean_v_chain[-1]

        return v_sample, mean_v, updates_inference

    ###############################
    ##       COST
    ###############################
    def cost_updates(self, optimizer):
        v_sample, mean_v, updates_train = self.inference(self.v_init, self.c_init)
        monitor_v = T.xlogx.xlogy0(self.v_init, mean_v)
        monitor = monitor_v.sum(axis=(1,2)) / self.temporal_order
        # Mean over batches
        monitor = T.mean(monitor)

        # Compute cost function
        fe_positive = self.free_energy(self.v_init, self.bv_dynamic, self.bh_dynamic)
        fe_negative = self.free_energy(v_sample, self.bv_dynamic, self.bh_dynamic)

        # Mean along batches
        cost = T.mean(fe_positive) - T.mean(fe_negative)

        # Weight decay
        cost = cost + self.weight_decay_coeff * self.get_weight_decay()

        # Update weights
        grads = T.grad(cost, self.params, consider_constant=[v_sample])
        updates_train = optimizer.get_updates(self.params, grads, updates_train)

        return cost, monitor, updates_train

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    def get_train_function(self, piano, orchestra, optimizer, name):

        super(cRnnRbm, self).get_train_function()

        # index to a [mini]batch : int32
        index = T.ivector()

        # get the cost and the gradient corresponding to one step of CD-15
        cost, monitor, updates = self.cost_updates(optimizer)

        return theano.function(inputs=[index],
                               outputs=[cost, monitor],
                               updates=updates,
                               givens={self.v_init: self.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.c_init: self.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_c)},
                               name=name
                               )

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        self.v_init = self.rng.uniform(low=0, high=1, size=(self.batch_size, self.temporal_order, self.n_v)).astype(theano.config.floatX)
        # Generate the last frame for the sequence v
        v_sample, _, updates_valid = self.inference(self.v_init, self.c_init)
        predicted_frame = v_sample[:,-1,:]
        # Get the ground truth
        true_frame = self.v_truth[:,-1,:]
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)

        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ###############################
    def get_validation_error(self, piano, orchestra, name):

        super(cRnnRbm, self).get_validation_error()

        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v_truth: self.build_sequence(orchestra, index, self.batch_size, self.temporal_order, self.n_v),
                                       self.c_init: self.build_sequence(piano, index, self.batch_size, self.temporal_order, self.n_c)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    #   Need no seed in this model
    ###############################
    def recurrence_generation(self, c_t, u_tm1):
        bv_t = self.bv + T.dot(u_tm1, self.Wuv) + T.dot(c_t, self.Wcv)
        bh_t = self.bh + T.dot(u_tm1, self.Wuh) + T.dot(c_t, self.Wch)

        # Orchestra initialization
        v_init_gen = self.rng.uniform(size=(self.batch_generation_size, self.n_v), low=0.0, high=1.0).astype(theano.config.floatX)

        # Dropout for RBM consists in applying the same mask to the hidden units at every the gibbs sampling step
        if self.step_flag == 'train':
            dropout_mask = self.rng.binomial(size=(self.batch_size, self.temporal_order, self.n_h), n=1, p=1-self.dropout_probability, dtype=theano.config.floatX)
        else:
            dropout_mask = (1-self.dropout_probability)

        # Inpainting :
        # p_t is clamped
        # perform k-step gibbs sampling to get o_t
        (_, v_chain), updates_inference = theano.scan(
            # Be careful argument order has been modified
            # to fit the theano function framework
            fn=lambda v,bv,bh: self.gibbs_step(v, bv, bh, dropout_mask),
            outputs_info=[None, v_init_gen],
            non_sequences=[bv_t, bh_t],
            n_steps=self.k
        )
        v_t = v_chain[-1]

        # update the rnn state
        u_t = T.tanh(self.bu + T.dot(v_t, self.Wvu) + T.dot(u_tm1, self.Wuu))

        return u_t, v_t, updates_inference

    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size,
                              batch_generation_size,
                              name="generate_sequence"):

        super(cRnnRbm, self).get_generate_function()

        # Seed_size is actually fixed by the temporal_order
        seed_size = self.temporal_order
        self.batch_generation_size = batch_generation_size

        ########################################################################
        #########       Debug Value
        self.c_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_c).astype(theano.config.floatX)
        self.v_seed.tag.test_value = self.rng_np.rand(batch_generation_size, seed_size, self.n_v).astype(theano.config.floatX)
        self.c_gen.tag.test_value = self.rng_np.rand(batch_generation_size, self.n_c).astype(theano.config.floatX)
        ########################################################################

        ########################################################################
        #########       Initial hidden recurrent state (theano function)
        # Infer the state u at the end of the seed sequence
        u0 = T.zeros((batch_generation_size, self.n_u))  # initial value for the RNN hidden
        #########
        u0.tag.test_value = np.zeros((batch_generation_size, self.n_u), dtype=theano.config.floatX)
        #########
        (u_chain), updates_initialization = self.rnn_inference(self.v_seed, self.c_seed, u0)
        u_seed = u_chain[-1]
        index = T.ivector()
        index.tag.test_value = [199, 1082]
        # Get the indices for the seed and generate sequences
        end_seed = index - generation_length + seed_size
        seed_function = theano.function(inputs=[index],
                                        outputs=[u_seed],
                                        updates=updates_initialization,
                                        givens={self.v_seed: self.build_sequence(orchestra, end_seed, batch_generation_size, seed_size, self.n_v),
                                                self.c_seed: self.build_sequence(piano, end_seed, batch_generation_size, seed_size, self.n_c)},
                                        name=name
                                        )
        ########################################################################

        ########################################################################
        #########       Next sample
        # Graph for the orchestra sample and next hidden state
        u_t, v_t, updates_next_sample = self.recurrence_generation(self.c_gen, self.u_gen)
        # Compile a function to get the next visible sample
        next_sample = theano.function(
            inputs=[self.c_gen, self.u_gen],
            outputs=[u_t, v_t],
            updates=updates_next_sample,
            name="next_sample",
        )
        ########################################################################

        def closure(ind):
            # Get the initial hidden chain state
            (u_t,) = seed_function(ind)

            # Initialize generation matrice
            piano_gen, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)

            for time_index in xrange(seed_size, generation_length, 1):
                # Build piano vector
                present_piano = piano_gen[:,time_index,:]
                # Next Sample and update hidden chain state
                u_t, o_t = next_sample(present_piano, u_t)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:,time_index,:] = o_t

            return (orchestra_gen,)

        return closure