#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Convert the matrices for binary, categorical or real units

from acidano.data_processing.utils.categorical import from_continuous_to_categorical
import numpy as np
import re

def type_conversion(dico_matrix, unit_type):
    result = {}
    # Binary unit ?
    if unit_type == 'binary':
        for k,matrix in dico_matrix.iteritems():
            matrix[np.nonzero(matrix)] = 1
            result[k] = matrix
    elif unit_type == 'continuous':
        # Much easier to work with unit between 0 and 1, for several reason :
        #       - 'normalized' values
        #       - same reconstruction as for binary units when building midi files
        for k,matrix in dico_matrix.iteritems():
            matrix = matrix / 127
            result[k] = matrix
    elif re.search('categorical', unit_type):
        # Categorical
        for k,matrix in dico_matrix.iteritems():
            m = re.search(r'[0-9]+$', unit_type)
            N_category = int(m.group(0))
            matrix = matrix / 127
            matrix = from_continuous_to_categorical(matrix, N_category)
            result[k] = matrix
    return result
