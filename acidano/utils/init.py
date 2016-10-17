#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano
import numpy as np
from numpy.random import RandomState


# Initialization functions
def shared_normal(num_rows, num_cols, scale=1, rng=None, name=None):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    if rng is None:
        rng = RandomState(seed=np.random.randint(1 << 30))
    return theano.shared(rng.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX),
        name=name)


def shared_zeros(shape, name=None):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                         name=name)


def castX(value):
    return theano._asarray(value, dtype=theano.config.floatX)


def sharedX(value, name=None, borrow=False):
    return theano.shared(castX(value), name=name, borrow=borrow)
