#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano.tensor as T
import numpy as np


####################################################################
####################################################################
# Binaries measures
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
####################################################################
####################################################################

####################################################################
####################################################################
# Real
def accuracy_measure_not_shared_continuous(true_frame, pred_frame):
    threshold = 0.05
    diff_abs = np.absolute(true_frame - pred_frame)
    diff_thresh = np.where(diff_abs < threshold, 1, 0)
    binary_truth = np.where(pred_frame > 0, 1, 0)
    tp = np.sum(diff_thresh * binary_truth, axis=-1)
    import pdb; pdb.set_trace()
    false = np.sum((1 - diff_thresh), axis=-1)

    quotient = tp + false

    accuracy_measure =  np.where(np.equal(quotient, 0), 0, np.true_divide(tp, quotient))

    return accuracy_measure
####################################################################
####################################################################

if __name__ == '__main__':
    t = np.asarray([[0.6,0.3,0.2,0], [0.6,0.3,0.2,0]])
    p = np.asarray([[0.64,0.2,0.4,0], [0.6,0.3,0.2,0]])
    accuracy_measure_not_shared_real(t,p)
