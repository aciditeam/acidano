#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from acidano.visualization.waveform.write_waveform_html import write_waveform_html

def visualize_waveform(waveform, path, file_name_no_extension):
    temp_csv = path + '/' + file_name_no_extension + '.csv'
    with open(temp_csv, 'wb') as f:
        f.write("x,y\n")
        for x in range(len(waveform)):
            f.write(str(x) + ',' + str(waveform[x]) + '\n')

    write_waveform_html(path + '/' + file_name_no_extension + ".html", file_name_no_extension)


if __name__ == '__main__':
    import numpy as np
    waveform=np.zeros((5000))
    for x in range(5000):
        waveform[x] = math.sin(2 * math.pi * x / 153)
    visualize_waveform(waveform, '.', 'test')
