#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Convert pianoroll to categorical representations
# 0 forms an isolated category

import math
import numpy as np


def from_continuous_to_categorical(pr_continuous, C):
    # pr_continuous = matrix, values between 0 1
    # C = number of category
    #
    # Output a pr with size (Time, Pitch * N_category)
    # For example, pitch = 2, Cat = 3 and T = 1, we have in this order the pairs (Pitch,Cat)
    # ((1,1), (1,2), (1,3), (2,1), (2,2), (2,3))
    #
    # Hence, units in the model have to be softmax over N_category (here 3) since one category can be on at each time

    pr_cat = np.zeros((pr_continuous.shape[0], pr_continuous.shape[1]*C))

    T = pr_continuous.shape[0]
    P = pr_continuous.shape[1]

    # Mutliply pr_continuous by the number of category
    for t in range(T):
        for p in range(P):
            if pr_continuous[t,p] == 0.0:
                cat_intensity = 0
            elif pr_continuous[t,p] == 1.0:
                cat_intensity = C-1
            else:
                cat_intensity = int(math.floor(pr_continuous[t,p] * (C-1)) + 1)
            ind_cat = p * C + cat_intensity
            pr_cat[t,ind_cat] = 1

    ########## DEBUG ######################
    # from acidano.visualization.numpy_array.visualize_numpy import visualize_mat
    # visualize_mat(pr_continuous, 'DEBUG', 'continuous')
    # visualize_mat(pr_cat, 'DEBUG', 'categorical')
    # import pdb; pdb.set_trace()
    #######################################

    return pr_cat
