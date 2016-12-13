#!/usr/bin/env python
# -*- coding: utf8 -*-

from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv
import os
import re

from acidano.utils import hopt_wrapper

import numpy as np
from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import theano.tensor as T
import theano

from hyperopt import hp
from math import log


class Model_lop(object):
    """
    Template class for the lop models.
    Contains plot methods
    """

    def __init__(self, model_param, dimensions):
        # Training parameters
        self.batch_size = dimensions['batch_size']
        self.temporal_order = dimensions['temporal_order']

        # Regularization paramters
        self.dropout_probability = model_param['dropout_probability']
        self.weight_decay_coeff = model_param['weight_decay_coeff']

        # Numpy and theano random generators
        self.rng_np = RandomState(25)
        self.rng = RandomStreams(seed=25)

        self.params = []
        self.step_flag = None
        return

    @staticmethod
    def get_hp_space():
        space_training = {'batch_size': hopt_wrapper.quniform_int('batch_size', 50, 500, 1),
                          'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(3), log(20), 1)
                          }

        space_regularization = {'dropout_probability': hp.choice('dropout', [
            0.0,
            hp.normal('dropout_probability', 0.5, 0.1)
        ]),
            'weight_decay_coeff': hp.choice('weight_decay_coeff', [
                0.0,
                hp.uniform('a', 1e-4, 1e-2)
            ])
        }

        space_training.update(space_regularization)
        return space_training

    def save_weights(self, save_folder):
        # Find acidano path
        # user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
        # NOT_FOUND = True
        # for user_path in user_paths:
        #     if re.search('acidano', user_path):
        #         d3js_source_path = user_path + 'acidano/visualization/d3.v3.min.js'
        #         NOT_FOUND = False
        #         break
        # if NOT_FOUND:
        #     print("No d3.js -> No visualization")
        #     return

        def plot_process(param_shared):
            param = param_shared.get_value()

            temp_csv = save_folder + '/' + param_shared.name + '.csv'
            np.savetxt(temp_csv, param, delimiter=',')

            # # Get mean, std and write title
            # mean = np.mean(param)
            # std = np.mean(param)
            # title = param_shared.name + " mean = " + str(mean) + " std = " + str(std)
            #
            # # Plot histogram
            # fig = plt.figure()
            # fig.suptitle(title, fontsize=14, fontweight='bold')
            #
            # ax = fig.add_subplot(111)
            # fig.subplots_adjust(top=0.85)
            #
            # ax.set_xlabel('nb_occurence')
            # ax.set_ylabel('value')
            #
            # param_ravel = param.ravel()
            # # Check for NaN values
            # if np.sum(np.isnan(param_ravel)):
            #     # Give an arbitrary value
            #     param_ravel = np.zeros(param_ravel.shape) - 1
            #     fig.suptitle(title + " NAN !!", fontsize=14, fontweight='bold')
            #
            # n, bins, patches = plt.hist(param_ravel, bins=50, normed=1, histtype='bar', rwidth=0.8)
            # plt.savefig(save_folder + '/' + param_shared.name + '.pdf')
            # plt.close(fig)
            #
            # # Plot matrices
            # temp_csv = save_folder + '/' + param_shared.name + '.csv'
            # np.savetxt(temp_csv, param, delimiter=',')
            # dump_to_csv(temp_csv, temp_csv)
            # write_numpy_array_html(save_folder + '/' + param_shared.name + '.html', param_shared.name, d3js_source_path)

        # Plot weights
        for param_shared in self.params:
            plot_process(param_shared)

    def get_weight_decay(self):
        ret = 0
        for param in self.params:
            ret += T.pow(param, 2).sum()
        return ret

    ###############################
    ##       Building matrices
    ###############################
    ######
    ## Sequential models
    ######
    def build_sequence(self, pr, index, batch_size, seq_length, last_dim):
        # [T-1, T-2, ..., 0]
        decreasing_time = theano.shared(np.arange(seq_length-1,-1,-1, dtype=np.int32))
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
        temporal_shift = T.tile(decreasing_time, (batch_size,1))
        # Reshape
        index_full = index.reshape((batch_size, 1)) - temporal_shift
        # Slicing
        pr = pr[index_full.ravel(),:]
        # Reshape
        return T.reshape(pr, (batch_size, seq_length, last_dim))

    ######
    ## Those functions are used for generating sequences
    ## with originally non-sequential models
    ## such as RBM, cRBM, FGcRBM...
    ######
    def build_seed(self, pr, index, batch_size, length_seq):
        n_dim = len(pr.shape)
        last_dim = pr.shape[n_dim-1]
        # [T-1, T-2, ..., 0]
        decreasing_time = np.arange(length_seq-1,-1,-1, dtype=np.int32)
        #
        temporal_shift = np.tile(decreasing_time, (batch_size,1))
        # Reshape
        index_full = index.reshape((batch_size, 1)) - temporal_shift
        # Slicing
        seed_pr = pr[index_full,:]\
            .ravel()\
            .reshape((batch_size, length_seq, last_dim))
        return seed_pr

    def initialization_generation(self, piano, orchestra, ind, generation_length, batch_generation_size, seed_size):
        # Build piano generation
        piano_gen = self.build_seed(piano.get_value(), ind,
                                    batch_generation_size, generation_length)

        # Build orchestra seed and cast it in the orchestration generation vector
        first_generated_ind = (ind - generation_length + seed_size) + 1
        last_orchestra_seed_ind = first_generated_ind - 1
        orchestra_seed = self.build_seed(orchestra.get_value(), last_orchestra_seed_ind,
                                         batch_generation_size, seed_size)

        n_orchestra = orchestra.get_value().shape[1]
        orchestra_gen = np.zeros((batch_generation_size, generation_length, n_orchestra)).astype(theano.config.floatX)
        orchestra_gen[:, :seed_size, :] = orchestra_seed
        return piano_gen, orchestra_gen

    ###############################
    ##       Getter for theano function
    ###############################
    def get_train_function(self):
        self.step_flag = 'train'

    def get_validation_error(self):
        self.step_flag = 'validate'

    def get_generate_function(self):
        self.step_flag = 'generate'
