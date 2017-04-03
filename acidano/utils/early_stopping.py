#!/usr/bin/env python
# -*- coding: utf8 -*-


def up_criterion(val_tab, epoch, number_strips=3, validation_order=2):
    #######################################
    # Article
    # Early stopping, but when ?
    # Lutz Prechelt
    # UP criterion
    UP = True
    OVERFITTING = True
    s = 0
    while(UP and s < number_strips):
        t = epoch - s
        tmk = epoch - s - validation_order
        UP = val_tab[t] > val_tab[tmk] * 1.001
        s = s + 1
        if not UP:
            OVERFITTING = False
    return OVERFITTING
