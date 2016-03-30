#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Hyperopt
Variational LSTM
"""
# Numpy
import numpy as np
# Theano
import theano
import theano.tensor as T
# Hyperopt
from hyperopt import hp
from math import log

from acidano.data_processing.load_data import load_data_seq_tvt
from acidano.models.Variational_LSTM.class_def import Variational_LSTM


# Define hyper-parameter search space
def get_header():
    return ['n_z',  # latent space
            'n_h', 'n_c',
            'temporal_order',  # = size of the batches
            'beta1', 'beta2', 'alpha'  # ADAM parameters
            'accuracy']


def get_hp_space():
    space = (hp.qloguniform('n_z', log(100), log(1000), 20),
             hp.qloguniform('n_h', log(100), log(1000), 20),
             hp.qloguniform('n_c', log(100), log(1000), 20),
             hp.qloguniform('temporal_order', log(10), log(100), 10),
             # hyper-parameter for Adam are from Kingma & Lei Ba
             hp.uniform('beta1', 0, 0.9),
             hp.uniform('beta2', 0.99, 0.999),
             hp.loguniform('alpha', -5, -1)
             )
    return space


def train(params, dataset, temporal_granularity, log_file_path):
    # Hyperparams
    n_z, n_h, n_c, temporal_order, beta1, beta2, alpha = params

    # Cast the hp
    n_z = int(n_z)
    n_h = int(n_h)
    n_c = int(n_c)
    temporal_order = int(temporal_order)

    # Optimizer is passed to the train function as  dictionary
    optimizer = {'name': 'adam_l2',
                 'alpha': alpha,
                 'beta1': beta1,
                 'beta2': beta2}

    # Log them
    with open(log_file_path, 'ab') as log_file:
        log_file.write((u'# n_z :  {}\n'.format(n_z)).encode('utf8'))
        log_file.write((u'# n_h :  {}\n'.format(n_h)).encode('utf8'))
        log_file.write((u'# n_c :  {}\n'.format(n_c)).encode('utf8'))
        log_file.write((u'# temporal_order :  {}\n'.format(temporal_order)).encode('utf8'))
        log_file.write((u'# beta1 :  {}\n'.format(beta1)).encode('utf8'))
        log_file.write((u'# beta2 :  {}\n'.format(beta2)).encode('utf8'))
        log_file.write((u'# alpha :  {}\n'.format(alpha)).encode('utf8'))
    # Print
    print((u'# n_z :  {}'.format(n_z)).encode('utf8'))
    print((u'# n_h :  {}'.format(n_h)).encode('utf8'))
    print((u'# n_c :  {}'.format(n_c)).encode('utf8'))
    print((u'# temporal_order :  {}'.format(temporal_order)).encode('utf8'))
    print((u'# beta1 :  {}'.format(beta1)).encode('utf8'))
    print((u'# beta2 :  {}'.format(beta2)).encode('utf8'))
    print((u'# alpha :  {}'.format(alpha)).encode('utf8'))

    ########################################
    # Load data
    ########################################
    # Dimension : time * pitch
    orch, orch_mapping, piano, piano_mapping, train_index, val_index, _ \
        = load_data_seq_tvt(data_path=dataset,
                            log_file_path='bullshit.txt',
                            temporal_granularity=temporal_granularity,
                            temporal_order=temporal_order,
                            shared_bool=True,
                            bin_unit_bool=True,
                            split=(0.7, 0.1, 0.2))

    # Get dimensions
    orch_dim = orch.get_value(borrow=True).shape[1]
    piano_dim = piano.get_value(borrow=True).shape[1]

    # allocate symbolic variables for the data
    index = T.lvector()             # index to a [mini]batch
    o = T.matrix('o')
    p = T.matrix('p')

    ########################################
    # Instantiate the Variational_LSTM class
    ########################################
    # Units dim dictionary
    units_dim = {}
    units_dim['piano_dim'] = piano_dim
    units_dim['orch_dim'] = orch_dim
    units_dim['h_dim'] = n_h
    units_dim['z_dim'] = n_z
    # Units dim dictionary
    reparametrization_dim = {}
    reparametrization_dim['p2h_dim'] = n_h
    reparametrization_dim['o2h_dim'] = n_h
    reparametrization_dim['z2h_dim'] = n_h
    reparametrization_dim['prior2z_dim'] = n_z
    # Units dim dictionary
    lstm_dim = {}
    lstm_dim['cell_dim'] = n_c

    model = Variational_LSTM(orch=o,          # sequences as Theano matrices
                             piano=p,         # sequences as Theano matrices
                             units_dim=units_dim,
                             reparametrization_dim=reparametrization_dim,
                             lstm_dim=lstm_dim,
                             weights=None,
                             optimizer=optimizer,
                             numpy_rng=None,
                             theano_rng=None
                             )

    ########################################
    # Get symbolic graphs
    ########################################
    # get the cost and the gradient corresponding to one step of CD-15
    monitor_train, updates_train = model.cost_updates()
    monitor_val, accuracy, updates_valid = model.validation()

    ########################################
    # Compile theano functions
    ########################################
    # the purpose of train_crbm is solely to update the CRBM parameters
    train_vlstm = theano.function(inputs=[index],
                                  outputs=[monitor_train],
                                  updates=updates_train,
                                  givens={o: orch[index],
                                          p: piano[index]},
                                  name='train_vlstm')

    validation_error = theano.function(inputs=[index],
                                       outputs=[monitor_val, accuracy],
                                       updates=updates_valid,
                                       givens={o: orch[index],
                                               p: piano[index]},
                                       name='validation_error')

    # Training step
    epoch = 0
    OVERFITTING = False
    val_order = 4
    val_tab = np.zeros(val_order)
    while (not OVERFITTING):
        # go through the training set
        train_cost_epoch = []
        for ind_batch in train_index:
            # Train
            this_monitor = train_vlstm(ind_batch)
            # Keep track of MONITORING cost
            nll_upper_bound = this_monitor['nll_upper_bound']
            train_cost_epoch += [nll_upper_bound]

        if (epoch % 5 == 0):
            # Validation
            acc_store = []
            nll_upper_bound_val, recon_term, kl_term, max_orch, mean_orch, min_orch, \
                max_recon_orch_bin, mean_recon_orch_bin, min_recon_orch_bin = (0,) * 9
            for ind_batch in val_index:
                monitor_val, acc = validation_error(ind_batch)
                acc_store += [acc]
                nll_upper_bound_val += monitor_val['nll_upper_bound_val']
                recon_term += monitor_val['recon_term']
                kl_term += monitor_val['kl_term']
                max_orch += monitor_val['max_orch']
                mean_orch += monitor_val['mean_orch']
                min_orch += monitor_val['min_orch']
                max_recon_orch_bin += monitor_val['max_recon_orch_bin']
                mean_recon_orch_bin += monitor_val['mean_recon_orch_bin']
                min_recon_orch_bin += monitor_val['min_recon_orch_bin']

            # Stop if validation error decreased over the last three validation
            # "FIFO" from the left
            val_tab[1:] = val_tab[0:-1]
            mean_nll_upper_bound = np.mean(nll_upper_bound_val)
            check_decrease = np.sum(mean_nll_upper_bound >= val_tab[1:])
            if check_decrease == 0:
                OVERFITTING = True
            val_tab[0] = mean_nll_upper_bound

            with open(log_file_path, 'ab') as log_file:
                log_file.write(("Epoch : {} , Nll upper bound train : {} , Nll upper bound val : {} , Recon term : {} , KL term : {}\n"
                               .format(epoch, np.mean(train_cost_epoch), np.mean(nll_upper_bound_val),
                                       np.mean(recon_term), np.mean(kl_term)
                                       ))
                               .encode('utf8'))
                log_file.write(("Accuracy : {}\n"
                               .format(np.mean(acc_store)))
                               .encode('utf8'))
                log_file.write(("Min o : {} , Mean o : {} , Max o : {}\n"
                               .format(np.mean(min_orch), np.mean(mean_orch), np.mean(max_orch)))
                               .encode('utf8'))
                log_file.write(("Min o_rec : {} , Mean o_rec : {} , Max o_rec : {}\n"
                               .format(np.mean(max_recon_orch_bin), np.mean(max_recon_orch_bin), np.mean(max_recon_orch_bin)))
                               .encode('utf8'))
            print(("Epoch : {} , Nll upper bound train : {} , Nll upper bound val : {} , Recon term : {} , KL term : {}"
                   .format(epoch, np.mean(train_cost_epoch), np.mean(nll_upper_bound_val),
                           np.mean(recon_term), np.mean(kl_term)
                           ))
                  .encode('utf8'))
            print(("Accuracy : {}"
                   .format(np.mean(acc_store)))
                  .encode('utf8'))
            print(("Min o : {} , Mean o : {} , Max o : {}"
                   .format(np.mean(min_orch), np.mean(mean_orch), np.mean(max_orch)))
                  .encode('utf8'))
            print(("Min o_rec : {} , Mean o_rec : {} , Max o_rec : {}"
                   .format(np.mean(max_recon_orch_bin), np.mean(max_recon_orch_bin), np.mean(max_recon_orch_bin)))
                  .encode('utf8'))

        epoch += 1

    score = -np.amin(val_tab)  # To fit in the general framework of hyperopt, we search for a max
    dico_res = dict(zip(get_header(), params))
    dico_res['accuracy': score]

    return score, dico_res


def create_past_vector(piano, orch, batch_size, delay, orch_dim):
    orch_reshape = T.reshape(orch, (batch_size, delay * orch_dim))
    past = T.concatenate((piano, orch_reshape), axis=1)
    return past
