#!/usr/bin/env python
# -*- coding: utf8 -*-

# Theano
import theano
import theano.tensor as T

# Hyperopt
from hyperopt import hp
from math import log


class Gradient_descent(object):
    def __init__(self, params):
        self.lr = params['lr']

    @staticmethod
    def get_hp_space():
        space = {'lr': hp.loguniform('lr', log(0.1), log(0.001))}
        return space

    @staticmethod
    def name():
        return "gradient_descent"

    def get_updates(self, params, grads, updates):
        for gparam, param in zip(grads, params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(self.lr, dtype=theano.config.floatX)

        return updates
