#!/usr/bin/env python
# -*- coding: utf8 -*-

from acidano.models.lop_keras.model_lop_keras import Model_lop_keras


class Repeat(Model_lop_keras):
    def __init__(self, model_param, dimensions):
        Model_lop_keras.__init__(self, model_param, dimensions)
        return

    @staticmethod
    def get_hp_space():
        super_space = Model_lop_keras.get_hp_space()
        space = {}

        space.update(super_space)
        return space

    @staticmethod
    def get_name():
        return "repeat"

    def build_model(self):
        self.model = None
        return

    def fit(self, orch_past, orch_t, piano_past, piano_t):
        return None

    def validate(self, orch_past, orch_t, piano_past, piano_t):
        return orch_past[:, -1, :]

    @staticmethod
    def get_static_config():
        model_space = {}
        model_space['batch_size'] = 200
        model_space['temporal_order'] = 1
        model_space['dropout_probability'] = 0
        model_space['weight_decay_coeff'] = 0
        return model_space
