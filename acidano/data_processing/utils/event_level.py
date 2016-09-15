#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from pianoroll_processing import sum_along_instru_dim


def get_event_ind(pr):
    # Return the list of new events
    THRESHOLD = 0.01
    pr_diff = np.sum(np.absolute(pr[0:-1, :] - pr[1:, :]), axis=1)
    pr_event = (pr_diff > THRESHOLD).nonzero()
    return np.concatenate(([0], (pr_event[0] + np.ones(pr_event[0].shape[0], dtype=np.int))))


def get_event_ind_dict(pr_dict):
    pr_flat = sum_along_instru_dim(pr_dict)
    return get_event_ind(pr_flat)
