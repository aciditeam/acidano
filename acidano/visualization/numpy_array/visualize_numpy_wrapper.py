#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv


def visualize_dict(pr, path, file_name_no_extension):
    AAA = sum_along_instru_dim(pr)
    temp_csv = path + '/' + file_name_no_extension + '.csv'
    np.savetxt(temp_csv, AAA, delimiter=',')
    dump_to_csv(temp_csv, temp_csv)
    write_numpy_array_html(path + '/' + file_name_no_extension + ".html", file_name_no_extension)


def visualize_mat(pr, path, file_name_no_extension):
    temp_csv = path + '/' + file_name_no_extension + '.csv'
    np.savetxt(temp_csv, pr, delimiter=',')
    dump_to_csv(temp_csv, temp_csv)
    write_numpy_array_html(path + '/' + file_name_no_extension + ".html", file_name_no_extension)
