#!/usr/bin/env python
# -*- coding: utf-8 -*-

from acidano.visualization.waveform.write_waveform_html import write_waveform_html

def visualize_waveform(waveform, path, file_name_no_extension, subsampling_before_plot=1):
    waveform_subsampled = waveform[0::subsampling_before_plot]
    temp_csv = path + '/' + file_name_no_extension + '.csv'
    with open(temp_csv, 'wb') as f:
        f.write("x,y\n")
        for x in range(len(waveform_subsampled)):
            f.write(str(x) + ',' + str(waveform_subsampled[x]) + '\n')

    write_waveform_html(path + '/' + file_name_no_extension + ".html", file_name_no_extension)


if __name__ == '__main__':
    import numpy as np

    wave = np.loadtxt('waveform.csv', delimiter=',')
    wave = wave[:,1]

    # Call visualize_waveform
    visualize_waveform(wave, '.', 'test', 1)
