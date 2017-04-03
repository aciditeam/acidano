#!/usr/bin/env python
# -*- coding: utf8 -*-

# Theano
import theano
import theano.tensor as T

# Hyperopt
from hyperopt import hp
from math import log


class Rmsprop(object):
    """RMSProp with nesterov momentum and gradient rescaling."""

    def __init__(self, params):
        self.lr = params['lr']
        self.rho = 0.9
        # self.rescale = 10.0
        self.epsilon = 1e-6

    @staticmethod
    def get_hp_space():
        space = {'lr': hp.loguniform('lr', log(0.01), log(0.0001))}
        return space

    @staticmethod
    def name():
        return "rms_prop"

    def get_updates(self, params, grads, updates):
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + self.epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - self.lr * g))
        return updates
