#!/usr/bin/env python
# -*- coding: utf8 -*-

# Based on Bengio's team implementation

# Numpy
import numpy as np
# Theano
import theano
import theano.tensor as T
import theano.gradient as G
from theano.tensor.shared_randomstreams import RandomStreams
# Utils
from acidano.utils.init import shared_normal, shared_zeros
from acidano.utils.forward import propup_linear, propup_sigmoid, propup_relu, propup_tanh, propup_softplus
# Optimize
from acidano.utils.optim import adam_L2
# Cost
from acidano.utils.cost import KLGaussianGaussian
# Performance measures
from acidano.utils.measure import accuracy_measure
# Misc
from collections import OrderedDict


class Variational_LSTM(object):
    '''Simple class to train an RNN-RBM from MIDI files and to generate sample
    sequences.'''

    def __init__(self,
                 orch=None,  # sequences as Theano matrices
                 piano=None,  # sequences as Theano matrices
                 units_dim=None,
                 reparametrization_dim=None,
                 lstm_dim=None,
                 weights=None,
                 optimizer=None,
                 numpy_rng=None,
                 theano_rng=None):
        '''Constructs and compiles Theano functions for training and sequence
        generation.
        n_hidden : integer
            Number of hidden units of the conditional RBMs.
        n_hidden_recurrent : integer
            Number of hidden units of the RNN.
        lr : float
            Learning rate
        r : (integer, integer) tuple
            Specifies the pitch range of the piano-roll in MIDI note numbers,
            including r[0] but not r[1], such that r[1]-r[0] is the number of
            visible units of the RBM at a given time step. The default (21,
            109) corresponds to the full range of piano (88 notes).
        dt : float
            Sampling period when converting the MIDI files into piano-rolls, or
            equivalently the time difference between consecutive time steps.'''

        # Random generators
        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(seed=42)
        self.theano_rng = theano_rng

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.orch = orch
        if not orch:
            self.orch = T.matrix('o')

        self.piano = piano
        if not piano:
            self.piano = T.matrix('p')

        # units_dim can't be None
        assert (units_dim is not None), "You should provide dimensions for the units in the net"
        piano_dim = units_dim['piano_dim']
        orch_dim = units_dim['orch_dim']
        h_dim = units_dim['h_dim']
        z_dim = units_dim['z_dim']
        self.h_dim = h_dim
        self.z_dim = z_dim

        if reparametrization_dim is None:
            p2h_dim = 200
            o2h_dim = 200
            z2h_dim = 200
        else:
            p2h_dim = reparametrization_dim['p2h_dim']
            o2h_dim = reparametrization_dim['o2h_dim']
            z2h_dim = reparametrization_dim['z2h_dim']

        if weights is None:
            # Reparametrization networks
            # piano network
            self.W_p1 = shared_normal(piano_dim, p2h_dim, 0.001)
            self.W_p2 = shared_normal(p2h_dim, p2h_dim, 0.001)
            self.W_p3 = shared_normal(p2h_dim, p2h_dim, 0.001)
            self.W_p4 = shared_normal(p2h_dim, p2h_dim, 0.001)
            self.b_p1 = shared_zeros(p2h_dim)
            self.b_p2 = shared_zeros(p2h_dim)
            self.b_p3 = shared_zeros(p2h_dim)
            self.b_p4 = shared_zeros(p2h_dim)

            # orchestra network
            self.W_o1 = shared_normal(orch_dim, o2h_dim, 0.001)
            self.W_o2 = shared_normal(o2h_dim, o2h_dim, 0.001)
            self.W_o3 = shared_normal(o2h_dim, o2h_dim, 0.001)
            self.W_o4 = shared_normal(o2h_dim, o2h_dim, 0.001)
            self.b_o1 = shared_zeros(o2h_dim)
            self.b_o2 = shared_zeros(o2h_dim)
            self.b_o3 = shared_zeros(o2h_dim)
            self.b_o4 = shared_zeros(o2h_dim)

            # latent network
            self.W_z1 = shared_normal(z_dim, z2h_dim, 0.001)
            self.W_z2 = shared_normal(z2h_dim, z2h_dim, 0.001)
            self.W_z3 = shared_normal(z2h_dim, z2h_dim, 0.001)
            self.W_z4 = shared_normal(z2h_dim, z2h_dim, 0.001)
            self.b_z1 = shared_zeros(z2h_dim)
            self.b_z2 = shared_zeros(z2h_dim)
            self.b_z3 = shared_zeros(z2h_dim)
            self.b_z4 = shared_zeros(z2h_dim)

            # Encoder (inference)
            enc_dim = o2h_dim + p2h_dim + h_dim
            self.W_enc1 = shared_normal(enc_dim, enc_dim, 0.001)
            self.W_enc2 = shared_normal(enc_dim, enc_dim, 0.001)
            self.W_enc3 = shared_normal(enc_dim, enc_dim, 0.001)
            self.W_enc4 = shared_normal(enc_dim, enc_dim, 0.001)
            self.W_enc_mu = shared_normal(enc_dim, z_dim, 0.001)
            self.W_enc_sig = shared_normal(enc_dim, z_dim, 0.001)
            self.b_enc1 = shared_zeros(enc_dim)
            self.b_enc2 = shared_zeros(enc_dim)
            self.b_enc3 = shared_zeros(enc_dim)
            self.b_enc4 = shared_zeros(enc_dim)
            self.b_enc_mu = shared_zeros(z_dim)
            self.b_enc_sig = shared_zeros(z_dim)

            # Decoder (generation)
            # prior
            prior_dim = p2h_dim + h_dim
            self.W_prior1 = shared_normal(prior_dim, prior_dim, 0.001)
            self.W_prior2 = shared_normal(prior_dim, prior_dim, 0.001)
            self.W_prior3 = shared_normal(prior_dim, prior_dim, 0.001)
            self.W_prior4 = shared_normal(prior_dim, prior_dim, 0.001)
            self.W_prior_mu = shared_normal(prior_dim, z_dim, 0.001)
            self.W_prior_sig = shared_normal(prior_dim, z_dim, 0.001)
            self.b_prior1 = shared_zeros(prior_dim)
            self.b_prior2 = shared_zeros(prior_dim)
            self.b_prior3 = shared_zeros(prior_dim)
            self.b_prior4 = shared_zeros(prior_dim)
            self.b_prior_mu = shared_zeros(z_dim)
            self.b_prior_sig = shared_zeros(z_dim)

            dec_dim = z2h_dim + p2h_dim + h_dim
            self.W_dec1 = shared_normal(dec_dim, dec_dim, 0.001)
            self.W_dec2 = shared_normal(dec_dim, dec_dim, 0.001)
            self.W_dec3 = shared_normal(dec_dim, dec_dim, 0.001)
            self.W_dec4 = shared_normal(dec_dim, dec_dim, 0.001)
            self.W_dec_bin = shared_normal(dec_dim, orch_dim, 0.001)
            self.b_dec1 = shared_zeros(dec_dim)
            self.b_dec2 = shared_zeros(dec_dim)
            self.b_dec3 = shared_zeros(dec_dim)
            self.b_dec4 = shared_zeros(dec_dim)
            self.b_dec_bin = shared_zeros(orch_dim)

            # Recurence function
            # LSTM
            # input gate
            LSTM_in_dim = o2h_dim + z2h_dim
            self.L_oi = shared_normal(LSTM_in_dim, h_dim, 0.001)
            self.L_hi = shared_normal(h_dim, h_dim, 0.001)
            self.b_i = shared_zeros(h_dim)
            # Internal cell
            self.L_oc = shared_normal(LSTM_in_dim, h_dim, 0.001)
            self.L_hc = shared_normal(h_dim, h_dim, 0.001)
            self.b_c = shared_zeros(h_dim)
            # Forget gate

            self.L_of = shared_normal(LSTM_in_dim, h_dim, 0.001)
            self.L_hf = shared_normal(h_dim, h_dim, 0.001)
            self.b_f = shared_zeros(h_dim)
            # Output
            # No L_cout... as in Theano tuto
            self.L_oout = shared_normal(LSTM_in_dim, h_dim, 0.001)
            self.L_hout = shared_normal(h_dim, h_dim, 0.001)
            self.b_out = shared_zeros(h_dim)
        else:
            # load weights
            #  - for special initialization
            #  - from a previously trained model
            self.W_p1, self.W_p2, self.W_p3, self.W_p4, \
                self.b_p1, self.b_p2, self.b_p3, self.b_p4, \
                self.W_o1, self.W_o2, self.W_o3, self.W_o4, \
                self.b_o1, self.b_o2, self.b_o3, self.b_o4, \
                self.W_z1, self.W_z2, self.W_z3, self.W_z4, \
                self.b_z1, self.b_z2, self.b_z3, self.b_z4, \
                self.W_enc1, self.W_enc2, self.W_enc3, self.W_enc4, \
                self.W_enc_mu, self.W_enc_sig, \
                self.b_enc1, self.b_enc2, self.b_enc3, self.b_enc4, \
                self.b_enc_mu, self.b_enc_sig, \
                self.W_prior1, self.W_prior2, self.W_prior3, self.W_prior4, \
                self.W_prior_mu, self.W_prior_sig, \
                self.b_prior1, self.b_prior2, self.b_prior3, self.b_prior4, \
                self.b_prior_mu, self.b_prior_sig, \
                self.W_dec1, self.W_dec2, self.W_dec3, self.W_dec4, \
                self.W_dec_bin, \
                self.b_dec1, self.b_dec2, self.b_dec3, self.b_dec4, \
                self.b_dec_bin, \
                self.L_oi, self.L_hi, self.b_i, \
                self.L_oc, self.L_hc, self.b_c, \
                self.L_of, self.L_hf, self.b_f, \
                self.L_oout, self.L_hout, self.b_out = weights

        # We don't use the same learning rate for the different parts of the network
        # Hence we group them in different variables
        self.params = self.W_p1, self.W_p2, self.W_p3, self.W_p4, \
            self.b_p1, self.b_p2, self.b_p3, self.b_p4, \
            self.W_o1, self.W_o2, self.W_o3, self.W_o4, \
            self.b_o1, self.b_o2, self.b_o3, self.b_o4, \
            self.W_z1, self.W_z2, self.W_z3, self.W_z4, \
            self.b_z1, self.b_z2, self.b_z3, self.b_z4, \
            self.W_enc1, self.W_enc2, self.W_enc3, self.W_enc4, \
            self.W_enc_mu, self.W_enc_sig, \
            self.b_enc1, self.b_enc2, self.b_enc3, self.b_enc4, \
            self.b_enc_mu, self.b_enc_sig, \
            self.W_prior1, self.W_prior2, self.W_prior3, self.W_prior4, \
            self.W_prior_mu, self.W_prior_sig, \
            self.b_prior1, self.b_prior2, self.b_prior3, self.b_prior4, \
            self.b_prior_mu, self.b_prior_sig, \
            self.W_dec1, self.W_dec2, self.W_dec3, self.W_dec4, \
            self.W_dec_bin, \
            self.b_dec1, self.b_dec2, self.b_dec3, self.b_dec4, \
            self.b_dec_bin, \
            self.L_oi, self.L_hi, self.b_i, \
            self.L_oc, self.L_hc, self.b_c, \
            self.L_of, self.L_hf, self.b_f, \
            self.L_oout, self.L_hout, self.b_out

        # Initialize the optimizer
        if optimizer is None:
            optimizer = {'name': 'adam_l2',
                         'alpha': 0.001,
                         'beta1': 0.9,
                         'beta2': 0.999}
            self.optimizer = adam_L2(optimizer)
        elif optimizer['name'] == 'adam_l2':
            optimizer['epsilon'] = 1e-8
            self.optimizer = adam_L2(optimizer)

        ####################################################################
        ####################################################################
        ####################################################################
        # Test values
        self.debug_mode = False
        if self.debug_mode:
            # Since we mix shared variables and test_value,
            # we absolutely need to use the same dimensions for the test_values
            self.num_batches_test = 3  # We can choose any value here
            self.orch.tag.test_value = np.random.rand(self.num_batches_test, orch_dim)
            self.piano.tag.test_value = np.random.rand(self.num_batches_test, piano_dim)
        ####################################################################
        ####################################################################
        ####################################################################

    def Gaussian_sample(self, mu, sig):

        epsilon = self.theano_rng.normal(size=(mu.shape),
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        import pdb; pdb.set_trace()
        theano.printing.pydotprint(z, outfile="aaa.png")
        return z

    def lstm_prop(self, o_t, c_tm1, h_tm1):
        # Input gate
        i = propup_sigmoid(T.concatenate([o_t, h_tm1]), T.concatenate([self.L_oi, self.L_hi]), self.b_i)
        # Forget gate
        f = propup_sigmoid(T.concatenate([o_t, h_tm1]), T.concatenate([self.L_of, self.L_hf]), self.b_f)
        # Cell update term
        c_tilde = propup_tanh(T.concatenate([o_t, h_tm1]), T.concatenate([self.L_oc, self.L_hc]), self.b_c)
        c_t = f * c_tm1 + i * c_tilde
        # Output gate
        o = propup_sigmoid(T.concatenate([o_t, h_tm1]), T.concatenate([self.L_oout, self.L_hout]), self.b_out)
        # h_t
        h_t = o * T.tanh(c_t)

        return h_t, c_t

    def inference(self, o, p):
        # Infering z_t sequence from orch_t and piano_t
        #   (under both prior AND q distribution)

        # Initialize h_0 and c_0 states
        # h_0 = shared_zeros(self.h_dim)
        # c_0 = shared_zeros(self.h_dim)
        h_0 = T.zeros((self.h_dim,))
        c_0 = T.zeros(self.h_dim)
        # Test values
        if self.debug_mode:
            h_0.tag.test_value = np.zeros((self.h_dim,), dtype=theano.config.floatX)
            c_0.tag.test_value = np.zeros((self.h_dim), dtype=theano.config.floatX)

        # Orch network
        o_1 = propup_linear(o, self.W_o1, self.b_o1)
        o_2 = propup_relu(o_1, self.W_o2, self.b_o2)
        o_3 = propup_relu(o_2, self.W_o3, self.b_o3)
        o_4 = propup_relu(o_3, self.W_o4, self.b_o4)

        # Piano network
        p_1 = propup_relu(p, self.W_p1, self.b_p1)
        p_2 = propup_relu(p_1, self.W_p2, self.b_p2)
        p_3 = propup_relu(p_2, self.W_p3, self.b_p3)
        p_4 = propup_relu(p_3, self.W_p4, self.b_p4)

        def inner_fn(o_t, p_t, h_tm1, c_tm1):
            # This inner function describes one step of the recurrent process
            # Prior
            input_prior = T.concatenate([p_t, h_tm1])
            prior_1_t = propup_relu(input_prior, self.W_prior1, self.b_prior1)
            prior_2_t = propup_relu(prior_1_t, self.W_prior2, self.b_prior2)
            prior_3_t = propup_relu(prior_2_t, self.W_prior3, self.b_prior3)
            prior_4_t = propup_relu(prior_3_t, self.W_prior4, self.b_prior4)
            prior_mu_t = propup_linear(prior_4_t, self.W_prior_mu, self.b_prior_mu)
            prior_sig_t = propup_softplus(prior_4_t, self.W_prior_sig, self.b_prior_sig)

            # Inference term
            input_enc = T.concatenate([o_t, p_t, h_tm1])
            enc_1_t = propup_relu(input_enc, self.W_enc1, self.b_enc1)
            enc_2_t = propup_relu(enc_1_t, self.W_enc2, self.b_enc2)
            enc_3_t = propup_relu(enc_2_t, self.W_enc3, self.b_enc3)
            enc_4_t = propup_relu(enc_3_t, self.W_enc4, self.b_enc4)
            enc_mu_t = propup_linear(enc_4_t, self.W_enc_mu, self.b_enc_mu)
            enc_sig_t = propup_softplus(enc_4_t, self.W_enc_sig, self.b_enc_sig)

            # Why not directly normal(prior_mu_t, prior_sig_t) ?
            # Because theano can't handle nodes as parameters of a RandomStream
            epsilon_t = self.theano_rng.normal(size=(self.z_dim,), avg=0., std=1., dtype=theano.config.floatX)
            z_t_2 = prior_mu_t + prior_sig_t * epsilon_t
            # # Prevent gradient optimization through z
            z_t = G.disconnected_grad(z_t_2)
            import pdb; pdb.set_trace()
            theano.printing.pydotprint(z_t, 'z_t.html')
            theano.printing.pydotprint(z_t_2, 'z_t_2.html')

            # Compute Z network
            z_1_t = propup_relu(z_t, self.W_z1, self.b_z1)
            z_2_t = propup_relu(z_1_t, self.W_z2, self.b_z2)
            z_3_t = propup_relu(z_2_t, self.W_z3, self.b_z3)
            z_4_t = propup_relu(z_3_t, self.W_z4, self.b_z4)

            # Compute new recurrent hidden state
            input_lstm = T.concatenate([o_t, z_4_t])
            h_t, c_t = self.lstm_prop(input_lstm, c_tm1, h_tm1)

            return h_t, c_t, enc_mu_t, enc_sig_t, prior_mu_t, prior_sig_t, z_4_t, z_t

        # Scan through input sequence
        ((h, c, enc_mu, enc_sig, prior_mu, prior_sig, z_4, z), updates) =\
            theano.scan(fn=inner_fn,
                        sequences=[o_4, p_4],
                        outputs_info=[h_0, c_0, None, None, None, None, None, None])

        # Reconstruction from inferred z_t
        # Can be performed after scanning, which is computationally more efficient
        input_dec = T.concatenate([z_4, p_4, h], axis=1)
        dec_1 = propup_relu(input_dec, self.W_dec1, self.b_dec1)
        dec_2 = propup_relu(dec_1, self.W_dec2, self.b_dec2)
        dec_3 = propup_relu(dec_2, self.W_dec3, self.b_dec3)
        dec_4 = propup_relu(dec_3, self.W_dec4, self.b_dec4)
        dec_bin = propup_sigmoid(dec_4, self.W_dec_bin, self.b_dec_bin)

        # We need :
        #   - prior : p(z_t) (w/ reparametrization trick, just pass mu and sigma)
        #   - approx inference : q(z_t|x_t)
        #   - reconstruction : p(x|z)
        return (enc_mu, enc_sig, prior_mu, prior_sig, dec_bin, z), updates

    def compute_nll_upper_bound(self, validation=False):
        #############
        # Inference
        (enc_mu, enc_sig, prior_mu, prior_sig, dec_bin, z), updates = \
            self.inference(self.orch, self.piano)

        #############
        # Cost
        recon = T.sum(-T.nnet.binary_crossentropy(self.orch, dec_bin), axis=1)
        # binary_crossentropy = nll for binary input. Sum along input dimension, mean along time (i.e. batch)
        # for real-valued units, use GaussianNLL
        kl = KLGaussianGaussian(enc_mu, enc_sig, prior_mu, prior_sig)
        # Mean over batches
        recon_term = recon.mean()
        kl_term = kl.mean()
        # neg log-lik upper bound
        cost = recon_term + kl_term

        if not validation:
            #############
            # Gradient
            import pdb; pdb.set_trace()
            gparams = G.grad(cost, self.params)
            #############
            # Updates
            updates_train = self.optimizer.get_updates(self.params, gparams, updates)
        else:
            updates_train = {}

        #############
        # Monitor training
        # We use, as in Bengio a dictionary
        monitor = OrderedDict()
        # does the upper bound decrease ?
        monitor['nll_upper_bound'] = cost

        if validation:
            # If validation, compute more monitoring values
            monitor['recon_term'] = recon_term
            monitor['kl_term'] = kl_term

            # Original values
            max_orch = self.orch.max()
            mean_orch = self.orch.mean()
            min_orch = self.orch.min()
            monitor['max_orch'] = max_orch
            monitor['mean_orch'] = mean_orch
            monitor['min_orch'] = min_orch

            # Reconstructed distribution
            max_recon_orch_bin = dec_bin.max()
            mean_recon_orch_bin = dec_bin.mean()
            min_recon_orch_bin = dec_bin.min()
            monitor['max_recon_orch_bin'] = max_recon_orch_bin
            monitor['mean_recon_orch_bin'] = mean_recon_orch_bin
            monitor['min_recon_orch_bin'] = min_recon_orch_bin

        # Cost is in monitor
        return monitor, updates_train

    def cost_updates(self):
        monitor, updates_train = self.compute_nll_upper_bound(validation=False)
        return monitor, updates_train

    def validation(self):
        # Validation = computing the nll upper bound on the validation set
        # So this function simply is a wrapper of cost_update function w/ no updates
        monitor, updates_valid = self.compute_nll_upper_bound(validation=True)
        # Generate an orchestral sequence
        o, dec_bin, prior_mu, prior_sig, updates = self.generate(self.piano)
        updates_valid.update(updates)
        # Compute the model accuracy
        # Note that this is really different from the accuracy for the cRBM, since it is evaluated on a sequence and
        # that the past orchestration is not re-initialized at each prediction
        accuracy = accuracy_measure(self.orch, o)
        return monitor, accuracy, updates_valid

    def generation(self):
        o, _, _, _, updates = self.generate(self.piano)
        return o, updates

    def generate(self, p):
        # Genreate an orchestral sequence given a piano sequence

        # Initialize h_0 and c_0 states
        h_0 = T.zeros((self.h_dim,))
        c_0 = T.zeros((self.h_dim,))

        # First, compute the piano parametrization
        # Piano network
        p_1 = propup_relu(p, self.W_p1, self.b_p1)
        p_2 = propup_relu(p_1, self.W_p2, self.b_p2)
        p_3 = propup_relu(p_2, self.W_p3, self.b_p3)
        p_4 = propup_relu(p_3, self.W_p4, self.b_p4)

        def inner_fn(p_t, h_tm1, c_tm1):
            # This inner function describes one step of the recurrent process
            # z from prior
            input_prior = T.concatenate([p_t, h_tm1])
            prior_1_t = propup_relu(input_prior, self.W_prior1, self.b_prior1)
            prior_2_t = propup_relu(prior_1_t, self.W_prior2, self.b_prior2)
            prior_3_t = propup_relu(prior_2_t, self.W_prior3, self.b_prior3)
            prior_4_t = propup_relu(prior_3_t, self.W_prior4, self.b_prior4)
            prior_mu_t = propup_linear(prior_4_t, self.W_prior_mu, self.b_prior_mu)
            prior_sig_t = propup_softplus(prior_4_t, self.W_prior_sig, self.b_prior_sig)

            # Why not directly normal(prior_mu_t, prior_sig_t) ?
            # Because theano can't handle nodes as parameters of a RandomStream :
            # It will try to optimize through the random stream
            epsilon_t = self.theano_rng.normal(size=(self.z_dim,), avg=0., std=1., dtype=theano.config.floatX)
            z_t = prior_mu_t + prior_sig_t * epsilon_t

            # z network
            z_1_t = propup_relu(z_t, self.W_z1, self.b_z1)
            z_2_t = propup_relu(z_1_t, self.W_z2, self.b_z2)
            z_3_t = propup_relu(z_2_t, self.W_z3, self.b_z3)
            z_4_t = propup_relu(z_3_t, self.W_z4, self.b_z4)

            # Decode x_t
            input_dec = T.concatenate([z_4_t, p_t, h_tm1])
            dec_1 = propup_relu(input_dec, self.W_dec1, self.b_dec1)
            dec_2 = propup_relu(dec_1, self.W_dec2, self.b_dec2)
            dec_3 = propup_relu(dec_2, self.W_dec3, self.b_dec3)
            dec_4 = propup_relu(dec_3, self.W_dec4, self.b_dec4)
            dec_bin_t = propup_sigmoid(dec_4, self.W_dec_bin_t, self.b_dec_bin)

            o_t = self.theano_rng.binomial(n=1, p=dec_bin_t)

            # Orch network
            o_1 = propup_relu(o_t, self.W_o1, self.b_o1)
            o_2 = propup_relu(o_1, self.W_o2, self.b_o2)
            o_3 = propup_relu(o_2, self.W_o3, self.b_o3)
            o_4 = propup_relu(o_3, self.W_o4, self.b_o4)

            # Compute new recurrent hidden state
            input_lstm = T.concatenate([o_4, z_4_t])
            h_t, c_t = self.lstm_prop(input_lstm, c_tm1, h_tm1)

            return h_t, c_t, o_t, dec_bin_t, prior_mu_t, prior_sig_t

        ((h, c, o, dec_bin, prior_mu, prior_sig), updates) =\
            theano.scan(fn=inner_fn,
                        sequences=[p_4],
                        outputs_info=[h_0, c_0, None, None, None, None])

        return o, dec_bin, prior_mu, prior_sig, updates
