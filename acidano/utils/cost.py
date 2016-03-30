#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano.tensor as T
import numpy as np


def GaussianNLL(y, mu, sig):
    """
    Gaussian negative log-likelihood
    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    # Expression ok :
    #   -log(p(x))
    # with p a gaussian
    # BUT NOT WITH TEST VALUES
    nll = 0.5 * T.sum(T.sqr(y - mu) / sig ** 2 + 2 * T.log(sig) +
                      T.log(2 * np.pi), axis=1)

    # Summed along time (i.e. mini-batch)
    return nll


def KLGaussianGaussian(mu1, sig1, mu2, sig2):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.
    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    kl = T.sum(0.5 * (2 * T.log(sig2)
               - 2 * T.log(sig1)
               + (sig1 ** 2 + (mu1 - mu2) ** 2) / sig2 ** 2
               - 1), axis=1)
    return kl
