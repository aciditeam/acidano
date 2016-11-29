#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano.tensor as T
import numpy as np
import math


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


def gaussian_likelihood_diagonal_variance(t, mu, sig, dim):
    """
    Gaussian Likelihood along first dimension
    Parameters
    ----------
    t   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    dim : First dimension of the target vector t
    """
    # First clip sig
    sig_clip = T.clip(sig, 1e-40, 1e40)

    # Since the variance matrix is diagonal, normalization term is easier to compute,
    # and calculus overflow can easily be prevente by first summing by 2*pi and taking square
    sig_time_2pi = T.sqrt(sig_clip * 2 * math.pi)

    #######################
    #######################
    # This is the problem... product goes to 0
    normalization_coeff = T.clip(T.prod(sig_time_2pi, axis=0), 1e-40, 1e40)
    #######################
    #######################

    # Once again, fact that sig is diagonal allows for simplifications :
    # term by term division instead of inverse matrix multiplication
    exp_term = (T.exp(- 0.5 * (t-mu) * (t-mu) / sig_clip).sum(axis=0))
    pdf = exp_term / normalization_coeff
    return pdf


def gaussian_likelihood_scalar(t, mu, sig):
    """
    1D-Gaussian Likelihood
    Parameters
    ----------
    t   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    normalization_coeff = T.sqrt(sig * 2 * math.pi)
    exp_term = T.exp(- 0.5 * (t-mu) * (t-mu) / sig)
    return exp_term / normalization_coeff


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
