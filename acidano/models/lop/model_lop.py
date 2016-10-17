#!/usr/bin/env python
# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv

import os
import re

import numpy as np

import theano.tensor as T
import theano


class Model_lop(object):
    """
    Template class for the lop models.
    Contains plot methods
    """

    def __init__(self):
        self.params = []
        return

    def weights_visualization(self, save_folder):
        # Find acidano path
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
        NOT_FOUND = True
        for user_path in user_paths:
            if re.search('acidano', user_path):
                d3js_source_path = user_path + 'acidano/visualization/d3.v3.min.js'
                NOT_FOUND = False
                break
        if NOT_FOUND:
            print("No d3.js -> No visualization")
            return

        # Plot weights
        for param_shared in self.params:
            param = param_shared.get_value()

            # Get mean, std and write title
            mean = np.mean(param)
            std = np.mean(param)
            title = param_shared.name + " mean = " + str(mean) + " std = " + str(std)

            # Plot histogram
            fig = plt.figure()
            fig.suptitle(title, fontsize=14, fontweight='bold')

            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.85)

            ax.set_xlabel('nb_occurence')
            ax.set_ylabel('value')

            param_ravel = param.ravel()
            # Check for NaN values
            if np.sum(np.isnan(param_ravel)):
                # Give an arbitrary value
                param_ravel = np.zeros(param_ravel.shape) - 1
                fig.suptitle(title + " NAN !!", fontsize=14, fontweight='bold')

            n, bins, patches = plt.hist(param_ravel, bins=50, normed=1, histtype='bar', rwidth=0.8)
            plt.savefig(save_folder + '/' + param_shared.name + '.pdf')
            plt.close(fig)

            # Plot matrices
            temp_csv = save_folder + '/' + param_shared.name + '.csv'
            np.savetxt(temp_csv, param, delimiter=',')
            dump_to_csv(temp_csv, temp_csv)
            write_numpy_array_html(save_folder + '/' + param_shared.name + '.html', param_shared.name, d3js_source_path)

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
