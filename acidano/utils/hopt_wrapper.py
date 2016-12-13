#!/usr/bin/env python
# -*- coding: utf8 -*-


from hyperopt.pyll_utils import validate_label
from hyperopt.pyll import scope

@validate_label
def qloguniform_int(label, *args, **kwargs):
    return scope.int(
        scope.hyperopt_param(label,
                             scope.qloguniform(*args, **kwargs)))

@validate_label
def quniform_int(label, *args, **kwargs):
    return scope.int(
        scope.hyperopt_param(label,
                             scope.quniform(*args, **kwargs)))
