from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adagrad

import numpy as np


def load_data_train(window_size=22050, shift=0.5, batch_size=100, temporal_order=50, path='../Data/data.p'):

    assert (shift <= 1) and (shift > 0)

    # Load dico
    with open(path, 'rb') as f:
        dico = np.load(f)

    # Get data
    data = dico['data']
    total_time = data.shape[0]

    # Build Overlapping windows
    start_track = dico['start_track']
#     for track_ind in
#
#     end_track = start_track.append(total_time)
#     end_track.remove(0)
#     valid_ind = []
#     for (start_ind, end_ind) in zip(start_track, end_track):
#         start_valid_ind = start_ind + window_size * (1 + shift * (temporal_order-1))
#         valid_ind.extend(range(start_valid_ind, end_ind))
#
#     # Overlapping windows
#     start_ind = valid_ind[0]
#     end_ind = valid_ind[0] + window_size
#     last_ind = valid_ind()
#     while(end_ind < )
#
#     # Split to form minibatches
#     n_valid_batch = len(valid_ind)
#     N_batches = min(10000, n_valid_batch/batch_size)
#     train_batch = []
#     for ind_batch in range(N_batches):
#         train_batch.append(valid_ind[ind_batch*batch_size:(ind_batch+1)*batch_size])
#
#     return data, train_batch
#
#
#
# def mlp_train(input_dim):
#     model = Sequential()
#     # Dropout on inputs
#     model.add(Dropout())
#     # Layer 1
#     model.add(Dense(32, input_dim=input_dim))
#     model.add(Activation('relu'))
#     model.add(Dropout())
#     # Output
#     model.add(Dense())
#     model.add(Activation('tanh'))
#
#     model.compile(optimizer=Adagrad(lr=0.01, epsilon=1e-06),
#                   loss='categorical_crossentropy',
#                   metrics=['reconstruction'])  # A définir
