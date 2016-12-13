#!/usr/bin/env python
# -*- coding: utf8 -*-

import hyperopt

print(hyperopt.version.version)
import hyperopt.pyll_utils as pyll_utils
from hyperopt.pyll import scope

if 'validate_label' in dir(pyll_utils):
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
else:
    def qloguniform_int(label, *args, **kwargs):
        return scope.int(
            scope.hyperopt_param(label,
                                 scope.qloguniform(*args, **kwargs)))

    def quniform_int(label, *args, **kwargs):
        return scope.int(
            scope.hyperopt_param(label,
                                 scope.quniform(*args, **kwargs)))


