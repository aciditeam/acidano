#!/usr/bin/env python
# -*- coding: utf8 -*-

# Plot lib
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from acidano.visualization.numpy_array.visualize_numpy import visualize_mat

import numpy as np


def save_weights(model, save_folder):
    def plot_process(param, name):
        # Get mean, std and write title
        mean = np.mean(param)
        std = np.mean(param)
        min_v = np.min(param)
        max_v = np.max(param)
        title = name + " mean = " + str(mean) + " std = " + str(std) +\
            "\nmin = " + str(min_v) + " max = " + str(max_v)

        # Plot histogram
        fig = plt.figure()
        fig.suptitle(title, fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)

        ax.set_xlabel('value')
        ax.set_ylabel('nb_occurence')

        param_ravel = param.ravel()
        # Check for NaN values
        if np.sum(np.isnan(param_ravel)):
            # Give an arbitrary value
            param_ravel = np.zeros(param_ravel.shape) - 1
            fig.suptitle(title + " NAN !!", fontsize=14, fontweight='bold')

        n, bins, patches = plt.hist(param_ravel.T, bins=50, normed=1, histtype='bar', rwidth=0.8)
        plt.savefig(save_folder + '/' + name + '.pdf')
        plt.close(fig)

        # D3js plot (heavy...)
        temp_csv = save_folder + '/' + name + '.csv'
        np.savetxt(temp_csv, param, delimiter=',')
        visualize_mat(param, save_folder, name)

        # Plot matrices
        # image_rgba changes the value of param, so rename it
        param_plot = param
        xdim = param_plot.shape[0]
        if len(param_plot.shape) == 1:
            param_plot = param_plot.reshape((xdim, 1))
        ydim = param_plot.shape[1]
        view = param_plot.view(dtype=np.uint8).reshape((xdim, ydim, 4))
        for i in range(xdim):
            for j in range(ydim):
                value = (param_plot[i][j] - min_v) / (max_v - min_v)
                view[i, j, 0] = int(255 * value)
                view[i, j, 1] = int(255 * value)
                view[i, j, 2] = int(255 * value)
                view[i, j, 3] = 255
        output_file(save_folder + '/' + name + '_bokeh.html')
        p = figure(title=name, x_range=(0, xdim), y_range=(0, ydim))
        p.image_rgba(image=[param_plot.T], x=[0], y=[0], dw=[xdim], dh=[ydim])
        save(p)

    # Plot weights
    for layer in model.layers:
        params = layer.get_weights()
        if len(params) > 0:
            name = layer.name
            for ind, param in enumerate(params):
                plot_process(param, name + '_' + str(ind))
