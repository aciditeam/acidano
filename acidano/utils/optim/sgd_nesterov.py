#!/usr/bin/env python
# -*- coding: utf8 -*-

# Theano
import theano

# Hyperopt
from hyperopt import hp
from math import log

# Misc and perso
import numpy as np


class Sgd_nesterov(object):
    def __init__(self, params):
        self.lr = params['lr']
        self.momentum = params['momentum']

    @staticmethod
    def get_hp_space():
        space = {'lr': hp.loguniform('lr', log(0.1), log(0.001)),
                 'momentum': hp.loguniform('momentum', log(0.9), log(0.999))}
        return space

    @staticmethod
    def name():
        return "sgd_nesterov"

    def get_updates(self, params, grads, updates):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = self.momentum * memory - self.lr * grad
            update2 = self.momentum * self.momentum * memory - (
                1 + self.momentum) * self.lr * grad
            updates[memory] = update
            updates[param] = param + update2
        return updates
