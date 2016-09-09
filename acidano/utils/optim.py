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
from acidano.utils.init import sharedX


class gradient_descent(object):
    def __init__(self, params):
        self.lr = params['lr']

    @staticmethod
    def get_hp_space():
        space = (hp.loguniform('lr', log(0.1), log(0.001)),)
        return space

    @staticmethod
    def get_param_dico(param):
        if param is None:
            learning_rate = 0.0
        else:
            learning_rate = param[0]  # Need the indexing to unpack the only value...

        optim_param = {
            'lr': learning_rate
        }

        return optim_param

    @staticmethod
    def name():
        return "gradient_descent"

    def get_updates(self, params, grads, updates):
        for gparam, param in zip(grads, params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(self.lr, dtype=theano.config.floatX)

        return updates


class adam_L2(object):
    def __init__(self, params):
        self.b1 = params['beta1']
        self.b2 = params['beta2']
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

    @staticmethod
    def name():
        return "adam_L2"

    def get_updates(self, params, grads, updates):
        """
        From cle (Junyoung Chung toolbox)
        """
        i = sharedX(0., 'counter')
        i_t = i + 1.
        b1_t = self.b1 ** i_t
        b2_t = self.b2 ** i_t

        for param, grad in zip(params, grads):
            m = sharedX(param.get_value() * 0.)
            # WOW PUtain
            # p.get_value ensure that at each iteration we keep the same node in the the graph, and not intialize a new one
            v = sharedX(param.get_value() * 0.)
            m_t = self.b1 * m + (1 - self.b1) * grad
            v_t = self.b2 * v + (1 - self.b2) * grad ** 2
            m_t_hat = m_t / (1. - b1_t)
            v_t_hat = v_t / (1. - b2_t)
            g_t = m_t_hat / (T.sqrt(v_t_hat) + self.epsilon)
            p_t = param - self.alpha * g_t
            updates[m] = m_t
            updates[v] = v_t
            updates[param] = p_t

        updates[i] = i_t
        return updates


class rmsprop(object):
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

    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
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
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class sgd_nesterov(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class sgd(object):
    # Only here for API conformity with other optimizers
    def __init__(self, params):
        pass

    def updates(self, params, grads, learning_rate):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - learning_rate * grad))
        return updates

    """
    Usage:
    grads = T.grad(cost, self.params)
    #opt = sgd_nesterov(self.params)
    opt = rmsprop(self.params)
    updates = opt.updates(self.params, grads,
                          learning_rate / np.cast['float32'](self.batch_size),
                          momentum)
    """
