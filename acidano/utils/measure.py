#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano.tensor as T
import numpy as np


def accuracy_measure(true_frame, pred_frame):
    # true_frame must be a matrix of binaries vectors
    # Sum along the last dimension (usually pitch or pixel intensity dimension)
    axis = true_frame.ndim - 1
    true_positive = T.sum(pred_frame * true_frame, axis=axis)
    false_negative = T.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = T.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive

    accuracy_measure = T.switch(T.eq(quotient, 0), 0, true_positive / quotient)
    # Rmq : avec ce switch, si on prédit correctement un silence, le score est de 0...
    # This has to be fixed.

    return accuracy_measure


def accuracy_measure_not_shared(true_frame, pred_frame):
    axis = len(true_frame.shape) - 1

    true_positive = np.sum(pred_frame * true_frame, axis=axis)
    false_negative = np.sum((1 - pred_frame) * true_frame, axis=axis)
    false_positive = np.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_negative + false_positive

    accuracy_measure = np.where(np.equal(quotient, 0), 0, np.true_divide(true_positive, quotient))

    return accuracy_measure


def recall_measure(true_frame, pred_frame):
    axis = true_frame.ndim - 1
    # true_frame must be a binary vector
    true_positive = T.sum(pred_frame * true_frame, axis=axis)
    false_negative = T.sum((1 - pred_frame) * true_frame, axis=axis)

    quotient = true_positive + false_negative

    recall_measure = T.switch(T.eq(quotient, 0), 0, true_positive / quotient)

    return recall_measure


def precision_measure(true_frame, pred_frame):
    axis = true_frame.ndim - 1
    # true_frame must be a binary vector
    true_positive = T.sum(pred_frame * true_frame, axis=axis)
    false_positive = T.sum(pred_frame * (1 - true_frame), axis=axis)

    quotient = true_positive + false_positive

    precision_measure = T.switch(T.eq(quotient, 0), 0, true_positive / quotient)

    return precision_measure


def cross_entropy(p, q):
    return T.xlogx.xlogy0(p, T.switch(T.eq(q, 0), 1e-20, q))
