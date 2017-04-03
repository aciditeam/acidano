#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# General remarks about the architecture
# Model is not part of the object, since we want to save it in a different adapted format
# Hence the object only conatins "meta information", not the model itself

from acidano.utils import hopt_wrapper
from hyperopt import hp
from math import log


class Model_lop_keras(object):
    """Template class for the lop keras models."""

    def __init__(self, model_param, dimensions):
        # Training parameters
        self.batch_size = model_param['batch_size']
        self.temporal_order = model_param['temporal_order']
        # Regularization paramters
        self.dropout_probability = model_param['dropout_probability']
        self.weight_decay_coeff = model_param['weight_decay_coeff']
        # Dimensions
        self.temporal_order = model_param['temporal_order']
        self.orch_dim = dimensions['orch_dim']
        self.piano_dim = dimensions['piano_dim']
        return

    @staticmethod
    def get_hp_space():
        space_training = {'batch_size': hopt_wrapper.quniform_int('batch_size', 50, 500, 1),
                          'temporal_order': hopt_wrapper.qloguniform_int('temporal_order', log(3), log(20), 1)
                          }
        space_regularization = {'dropout_probability': hp.choice('dropout', [
            0.0,
            hp.normal('dropout_probability', 0.5, 0.1)
        ]),
            'weight_decay_coeff': hp.choice('weight_decay_coeff', [
                0.0,
                hp.uniform('a', 1e-4, 1e-4)
            ])
        }
        space_training.update(space_regularization)
        return space_training
