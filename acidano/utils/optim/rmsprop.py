#!/usr/bin/env python
# -*- coding: utf8 -*-

# Theano
import theano
import theano.tensor as T

# Hyperopt
from hyperopt import hp
from math import log

# Misc and perso
import numpy as np

class Rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

        self.lr = params['lr']
        self.m = 0.9
        self.rescale = 5.0

    @staticmethod
    def get_hp_space():
        space = {'lr': hp.loguniform('lr', log(0.01), log(0.0001))}
        return space

    @staticmethod
    def name():
        return "gradient_descent"

    def updates(self, params, grads, updates):
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = self.rescale
        scaling_den = T.maximum(self.rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = self.m * memory - self.lr * grad / rms_grad
            update2 = self.m * self.m * memory - (
                1 + self.m) * self.lr * grad / rms_grad
            updates[old_square] = new_square
            updates[old_avg] = new_avg
            updates[memory] = update
            updates[param] = (param + update2)
        return updates
