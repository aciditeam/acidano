#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv


class Model_lop(object):
    """
    Template class for the lop models.
    Contains plot methods
    """

    def __init__(self):
        self.params = []
        return

    def weights_visualization(self, save_folder):
        for param_shared in self.params:
            param = param_shared.get_value()

            # Get mean, std and write title
            mean = np.mean(param)
            std = np.mean(param)
            title = param_shared.name + " mean = " + str(mean) + " std = " + str(std)

            # Plot histogram
            fig = plt.figure()
            fig.suptitle(title, fontsize=14, fontweight='bold')

            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.85)

            ax.set_xlabel('nb_occurence')
            ax.set_ylabel('value')

            param_ravel = param.ravel()
            # Check for NaN values
            if np.sum(np.isnan(param_ravel)):
                # Give an arbitrary value
                param_ravel = np.zeros(param_ravel.shape) - 1
                fig.suptitle(title + " NAN !!", fontsize=14, fontweight='bold')

            n, bins, patches = plt.hist(param_ravel, bins=50, normed=1, histtype='bar', rwidth=0.8)
            plt.savefig(save_folder + '/' + param_shared.name + '.pdf')
            plt.close(fig)

            # Plot matrices
            temp_csv = save_folder + '/' + param_shared.name + '.csv'
            np.savetxt(temp_csv, param, delimiter=',')
            dump_to_csv(temp_csv, temp_csv)
            write_numpy_array_html(save_folder + '/' + param_shared.name + '.html', param_shared.name)
