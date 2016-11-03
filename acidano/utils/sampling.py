#!/usr/bin/env python
# -*- coding: utf8 -*-


# Sampling function for theano graphs
def gaussian_sample(theano_rng, mu, sig):
    epsilon = theano_rng.normal(size=(mu.shape),
                                avg=0., std=1.,
                                dtype=mu.dtype)
    z = mu + sig * epsilon
    return z
