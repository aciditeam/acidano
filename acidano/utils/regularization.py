#!/usr/bin/env python
# -*- coding: utf8 -*-


import theano
import theano.tensor as T
from acidano.utils.init import shared_normal, shared_zeros

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# Sampling function for theano graphs
def dropout_function(data, p_dropout, size, rng=RandomStreams()):
    # Mask
    mask = rng.binomial(size=size, n=1, p=1-p_dropout, dtype=theano.config.floatX)

    # Apply mask and return
    return T.switch(mask, data, 0)


def batch_norm(matrix, dimensions):
    # # Instanciate weights
    # gamma = shared_normal(dimensions, 0.01, name='W_norm')
    # beta = shared_zeros(dimensions, name='b_norm')
    # Constants
    epsilon = 1e-6
    batch_dim = 0
    # Normalize
    mean = matrix.mean(axis=batch_dim, keepdims=True)
    var = matrix.var(axis=batch_dim, keepdims=True)
    x_hat = (matrix - mean) / T.sqrt(var + epsilon)

    # # Scale
    # y = gamma * x_hat + beta
    return x_hat


# def number_note_normalization_fun(matrix):
#     if number_note_normalization:
#         norm = matrix.sum(axis=-1, keepdims=True)
#         # Prevent division by zero
#         norm_switch = T.switch(T.eq(norm, 0), 1, norm)
#         return matrix / norm_switch
#
#     else:
#         return matrix
