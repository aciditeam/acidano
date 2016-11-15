#!/usr/bin/env python
# -*- coding: utf8 -*-


import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Sampling function for theano graphs
def dropout_function(data, p_dropout, size, rng=RandomStreams()):
    # Mask
    mask = rng.binomial(size=size, n=1, p=1-p_dropout, dtype=theano.config.floatX)

    # Apply mask and return
    return T.switch(mask, data, 0)
