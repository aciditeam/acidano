#!/usr/bin/env python
# -*- coding: utf8 -*-

""" Random for binary units """

# Model lop
from acidano.models.lop.model_lop import Model_lop

# Hyperopt
from acidano.utils import hopt_wrapper
from math import log

# Numpy
import numpy as np

# Theano
import theano
import theano.tensor as T

# Performance measures
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.measure import accuracy_measure, precision_measure, recall_measure


class Random(Model_lop):
    """Random : bernouilli(p)"""
    def __init__(self,
                 model_param,
                 dimensions,
                 checksum_database,
                 weights_initialization=None):

        super(Random, self).__init__(model_param, dimensions, checksum_database)
        self.n_orchestra = dimensions['orchestra_dim']
        self.n_piano = dimensions['piano_dim']

        # p should set equal to the average number of note on in the database
        self.p = model_param['p']

        self.v_truth = T.matrix('v_truth', dtype=theano.config.floatX)
        return

    ###############################
    ##       STATIC METHODS
    ##       FOR METADATA AND HPARAMS
    ###############################
    @staticmethod
    def get_hp_space():
        super_space = Model_lop.get_hp_space()
        space = {'p': hopt_wrapper.quniform_int('p', 50, 500, 1),
                 'temporal_order': hopt_wrapper.quniform_int('p', 1, 1, 1)}
        space.update(super_space)
        return space

    @staticmethod
    def name():
        return "Random bernouilli(p)"

    ###############################
    ##       TRAIN FUNCTION
    ###############################
    @Model_lop.train_flag
    def get_train_function(self, piano, orchestra, optimizer, name):
        def unit(i):
            a = 0
            b = 0
            return a, b
        return unit

    ###############################
    ##       PREDICTION
    ###############################
    def prediction_measure(self):
        predicted_frame = self.rng.binomial(size=(self.batch_size, self.n_orchestra), n=1, p=self.p)
        # Get the ground truth
        true_frame = self.v_truth
        # Measure the performances
        precision = precision_measure(true_frame, predicted_frame)
        recall = recall_measure(true_frame, predicted_frame)
        accuracy = accuracy_measure(true_frame, predicted_frame)
        updates_valid = {}
        return precision, recall, accuracy, updates_valid

    ###############################
    ##       VALIDATION FUNCTION
    ##############################
    def build_visible(self, orchestra, index):
        visible = orchestra[index,:]
        return visible
        
    @Model_lop.validation_flag
    def get_validation_error(self, piano, orchestra, name):
        # index to a [mini]batch : int32
        index = T.ivector()

        precision, recall, accuracy, updates_valid = self.prediction_measure()

        # This time self.v is initialized randomly
        return theano.function(inputs=[index],
                               outputs=[precision, recall, accuracy],
                               updates=updates_valid,
                               givens={self.v_truth: self.build_visible(orchestra, index)},
                               name=name
                               )

    ###############################
    ##       GENERATION
    ###############################
    @Model_lop.generate_flag
    def get_generate_function(self, piano, orchestra,
                              generation_length, seed_size, batch_generation_size,
                              name="generate_sequence"):
        def closure(ind):
            # Initialize generation matrice
            _, orchestra_gen = self.initialization_generation(piano, orchestra, ind, generation_length, batch_generation_size, seed_size)
            for index in xrange(seed_size, generation_length, 1):

                # Get the next sample
                v_t = np.random.binomial(n=1, p=self.p, size=(batch_generation_size, self.n_orchestra)).astype(theano.config.floatX)
                # Add this visible sample to the generated orchestra
                orchestra_gen[:,index,:] = v_t
            return (orchestra_gen,)

        return closure
