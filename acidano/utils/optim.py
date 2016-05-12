#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano
import theano.tensor as T
from acidano.utils.init import sharedX


class gradient_descent(object):
    def __init__(self, params):
        self.lr = params['lr']

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
