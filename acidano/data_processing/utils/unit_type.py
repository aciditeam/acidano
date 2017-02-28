#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Convert the matrices for binary, categorical or real units

from acidano.data_processing.utils.categorical import from_continuous_to_categorical
import numpy as np
import re

def type_conversion(matrix, unit_type):
    # Binary unit ?
    if unit_type == 'binary':
        matrix[np.nonzero(matrix)] = 1
    elif unit_type == 'continuous':
        # Much easier to work with unit between 0 and 1, for several reason :
        #       - 'normalized' values
        #       - same reconstruction as for binary units when building midi files
        matrix = matrix / 127
    elif re.search('categorical', unit_type):
        # Categorical
        m = re.search(r'[0-9]+$', unit_type)
        N_category = int(m.group(0))
        matrix = matrix / 127
        matrix = from_continuous_to_categorical(matrix, N_category)

    return matrix
