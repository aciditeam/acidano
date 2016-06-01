#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script is just a wrapper for visualizing
# a numpy array which has been dumped in a csv file
#
# Use :
#   python numpy_csv_visualization name_of_the_csv_file.csv

import sys
import os
import shutil
from subprocess import call
from sys import platform as _platform
import acidano.visualization.numpy_array.dumped_numpy_to_csv as dtc

MAIN_DIR = os.getcwd().decode('utf8') + u'/'

HTML_DIR = MAIN_DIR + u'visualization/numpy_array/'

path_to_csv = MAIN_DIR + sys.argv[1]  # Relative path to the CSV file containing the data
local_file = MAIN_DIR + 'Visualization/numpy_array/dump.csv'
shutil.copy(path_to_csv, local_file)

dtc.dump_to_csv(path_to_data=local_file, save_path=HTML_DIR + u'data.csv')

if _platform == "linux" or _platform == "linux2":
    # Linux
    call(["firefox", HTML_DIR + "numpy_vis.html"])
elif _platform == "darwin":
    # OS X
    call(["open", HTML_DIR + "numpy_vis.html"])
