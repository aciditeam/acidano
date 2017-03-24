#!/usr/bin/env python
# -*- coding: utf8 -*-

from acidano.models.lop_keras.model_lop_keras import Model_lop_keras

# Hyperopt
from hyperopt import hp
from acidano.utils import hopt_wrapper
from math import log

# Keras
import keras
from keras.layers import Input, LSTM, Dense
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
        # Stack lstm
        if self.n_hs:
            x = LSTM(self.n_hs[0], return_sequences=True, input_shape=(self.temporal_order, self.orch_dim),
                     dropout=self.dropout_probability,
                     kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                     bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(main_input)
            for l in range(1, len(self.n_hs)):
                x = LSTM(self.n_hs[l], return_sequences=True,
                         dropout=self.dropout_probability,
                         kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                         bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(x)
            # Seems logic to me that end of lstm has the same size as piano piano-roll
            lstm_out = LSTM(self.piano_dim,
                            dropout=self.dropout_probability,
                            kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                            bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(x)
        else:
            lstm_out = LSTM(self.piano_dim, input_shape=(self.temporal_order, self.orch_dim),
                            dropout=self.dropout_probability,
                            kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                            bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(main_input)
        # Auxiliary input
        auxiliary_input = Input(shape=(self.piano_dim,), name='piano_t')
        # Concatenate
        x = keras.layers.concatenate([lstm_out, auxiliary_input], axis=1)
        # Dense layers on top
        if self.binary:
            activation_top = 'sigmoid'
        else:
            activation_top = 'relu'
        orch_prediction = Dense(self.orch_dim, activation=activation_top, name='orch_pred',
                                kernel_regularizer=keras.regularizers.l2(self.weight_decay_coeff),
                                bias_regularizer=keras.regularizers.l2(self.weight_decay_coeff))(x)
        model = Model(inputs=[main_input, auxiliary_input], outputs=orch_prediction)
        # Instanciate the model
        self.model = model
        return

    def set_model(self, model):
        # Used to re-instanciate an already trained model
        self.model = model

    def fit(self, orch_past, orch_t, piano_past, piano_t):
        return (self.model).fit(x={'orch_seq': orch_past, 'piano_t': piano_t},
                                y=orch_t,
                                epochs=1,
                                batch_size=self.batch_size,
                                verbose=0)

    def validate(self, orch_past, orch_t, piano_past, piano_t):
        return (self.model).predict(x={'orch_seq': orch_past, 'piano_t': piano_t},
                                    batch_size=self.batch_size)
