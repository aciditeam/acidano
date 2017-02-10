#!/usr/bin/env python
# -*- coding: utf8 -*-

# Theano
import theano.tensor as T

# Hyperopt
from hyperopt import hp
from math import log

# Misc and perso
import theano
import numpy as np


class Adam_L2(object):
    def __init__(self, params):
        self.b1 = 0.9
        self.b2 = 0.999
        self.lr = params['lr']
        self.e = 1e-8

    @staticmethod
    def get_hp_space():
        space = {'lr': hp.loguniform('lr', log(0.001), log(0.001))}
        return space

    @staticmethod
    def name():
        return "adam_L2"

    """
    The MIT License (MIT)
    Copyright (c) 2015 Alec Radford
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def get_updates(self, params, grads, updates):
        i = theano.shared(np.float32(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - self.b1)**i_t
        fix2 = 1. - (1. - self.b2)**i_t
        lr_t = self.lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (self.b1 * g) + ((1. - self.b1) * m)
            v_t = (self.b2 * T.sqr(g)) + ((1. - self.b2) * v)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            p_t = p - (lr_t * g_t)
            updates[m] = m_t
            updates[v] = v_t
            updates[p] = p_t
        updates[i] = i_t
        return updates
