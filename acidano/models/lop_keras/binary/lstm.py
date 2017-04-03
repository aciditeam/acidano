#!/usr/bin/env python
# -*- coding: utf8 -*-

from acidano.models.lop_keras.model_lop_keras import Model_lop_keras

# Hyperopt
from hyperopt import hp
from acidano.utils import hopt_wrapper
from math import log

# Keras
import keras
from keras.layers import Input, LSTM, Dense, GRU
from keras.models import Model


class Lstm(Model_lop_keras):
    def __init__(self, model_param, dimensions):
        Model_lop_keras.__init__(self, model_param, dimensions)
        self.n_hs = model_param['n_hidden']
        self.binary = model_param['binary']
        return

    @staticmethod
    def get_hp_space():
        super_space = Model_lop_keras.get_hp_space()

        space = {'n_hidden': hp.choice('n_hidden', [
            [],
            [hopt_wrapper.qloguniform_int('n_hidden_1_'+str(i), log(100), log(5000), 10) for i in range(1)],
            [hopt_wrapper.qloguniform_int('n_hidden_2_'+str(i), log(100), log(5000), 10) for i in range(2)],
            [hopt_wrapper.qloguniform_int('n_hidden_3_'+str(i), log(100), log(5000), 10) for i in range(3)],
        ]),
        }

        space.update(super_space)
        return space

    @staticmethod
    def get_name():
        return "lstm_piano_pluged_last"

    def build_model(self):
        # Main input is a sequence of orchestra vectors
        main_input = Input(shape=(self.temporal_order, self.orch_dim), name='orch_seq')
        #####################
        # stacked LSTMs
        # First layer
        if len(self.n_hs) > 1:
            return_sequences = True
        else:
            return_sequences = False
        x = GRU(self.n_hs[0], return_sequences=return_sequences, input_shape=(self.temporal_order, self.orch_dim),
                dropout=self.dropout_probability,
                kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(main_input)
        if len(self.n_hs) > 1:
            # Intermediates layers
            for layer_ind in range(1, len(self.n_hs)):
                # Last layer ?
                if layer_ind == len(self.n_hs)-1:
                    return_sequences = False
                else:
                    return_sequences = True
                x = GRU(self.n_hs[layer_ind], return_sequences=return_sequences,
                        dropout=self.dropout_probability,
                        kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                        bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(x)
        lstm_out = x
        #####################
        # Auxiliary input
        auxiliary_input = Input(shape=(self.piano_dim,), name='piano_t')
        # Concatenate
        top_input = keras.layers.concatenate([lstm_out, auxiliary_input], axis=1)
        # Dense layers on top
        if self.binary:
            activation_top = 'sigmoid'
        else:
            activation_top = 'relu'
        orch_prediction = Dense(self.orch_dim, activation=activation_top, name='orch_pred',
                                kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                                bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(top_input)
        model = Model(inputs=[main_input, auxiliary_input], outputs=orch_prediction)
        # Instanciate the model
        self.model = model
        return

    def fit(self, orch_past, orch_t, piano_past, piano_t):
        return (self.model).fit(x={'orch_seq': orch_past, 'piano_t': piano_t},
                                y=orch_t,
                                epochs=1,
                                batch_size=self.batch_size,
                                verbose=0)

    def validate(self, orch_past, orch_t, piano_past, piano_t):
        return (self.model).predict(x={'orch_seq': orch_past, 'piano_t': piano_t},
                                    batch_size=self.batch_size)

    @staticmethod
    def get_static_config():
        model_space = {}
        model_space['batch_size'] = 200
        model_space['temporal_order'] = 10
        model_space['dropout_probability'] = 0
        model_space['weight_decay_coeff'] = 0
        # Last layer could be of size piano = 93
        model_space['n_hidden'] = [500, 500, 93]
        return model_space
