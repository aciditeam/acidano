#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np

# Test, let's dump a Nan array and an inf array
a = np.empty((3, 3,))
filename = "nan.csv"
np.savetxt(filename, a, delimiter=",")

b = np.array([[np.inf, np.inf], [np.inf, 1]])
filename = "inf.csv"
np.savetxt(filename, b, delimiter=",")
