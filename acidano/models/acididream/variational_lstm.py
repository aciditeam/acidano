#!/usr/bin/env python
# -*- coding: utf8 -*-

import os
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


class LSTM(object):
    '''LSTM for sequence generation
    This is a first "simple" version, based on http://arxiv.org/pdf/1412.6581v6.pdf
    Encoder/Decoder scheme, where the latent unit encode a whole sequence
    '''

    def __init__(self,
                 v=None,  # sequences as Theano matrices
                 units_dim=None,
                 weights=None,
                 optimizer=None,
                 numpy_rng=None,
                 theano_rng=None,
                 orch_debug=None,
                 piano_debug=None):
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

        self.v = v
        if not v:
            self.v = T.tensor3('v')

        # units_dim can't be None
        assert (units_dim is not None), "You should provide dimensions for the units in the net"
        v_dim = units_dim['v_dim']
        h_dim = units_dim['h_dim']
        z_dim = units_dim['z_dim']
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Encoder
        # LSTM
        # input gate
        self.L_vie = shared_normal(v_dim, h_dim, 0.01)
        self.L_hie = shared_normal(h_dim, h_dim, 0.01)
        self.b_ie = shared_zeros(h_dim)
        # Internal cell
        self.L_vce = shared_normal(v_dim, h_dim, 0.01)
        self.L_hce = shared_normal(h_dim, h_dim, 0.01)
        self.b_ce = shared_zeros(h_dim)
        # Forget gate
        self.L_vfe = shared_normal(v_dim, h_dim, 0.01)
        self.L_hfe = shared_normal(h_dim, h_dim, 0.01)
        self.b_fe = shared_zeros(h_dim)
        # Output
        # No L_cout... as in Theano tuto
        self.L_voute = shared_normal(v_dim, h_dim, 0.01)
        self.L_houte = shared_normal(h_dim, h_dim, 0.01)
        self.b_oute = shared_zeros(h_dim)
        # latent (parametrized through mu (mean) and sigma (var))
        self.W_hmu = shared_normal(h_dim, z_dim)
        self.W_hsig = shared_normal(h_dim, z_dim)
        self.b_mu = shared_normal(z_dim)
        self.b_sig = shared_normal(z_dim)

        # Decoder
        # latent
        self.W_zh = shared_normal(z_dim, h_dim)
        self.b_h0 = shared_zeros(h_dim)
        # recurrent
        # LSTM
        # input gate
        self.L_vid = shared_normal(v_dim, h_dim, 0.01)  # v actually refers to vt-1 when computing ht
        self.L_hid = shared_normal(h_dim, h_dim, 0.01)
        self.b_id = shared_zeros(h_dim)
        # Internal cell
        self.L_vcd = shared_normal(v_dim, h_dim, 0.01)
        self.L_hcd = shared_normal(h_dim, h_dim, 0.01)
        self.b_cd = shared_zeros(h_dim)
        # Forget gate
        self.L_vfd = shared_normal(v_dim, h_dim, 0.01)
        self.L_hfd = shared_normal(h_dim, h_dim, 0.01)
        self.b_fd = shared_zeros(h_dim)
        # Output
        # No L_cout... as in Theano tuto
        self.L_voutd = shared_normal(v_dim, h_dim, 0.01)
        self.L_houtd = shared_normal(h_dim, h_dim, 0.01)
        self.b_outd = shared_zeros(h_dim)
        # hidden -> visible
        self.W_hvo = shared_normal(h_dim, v_dim)
        self.b_vo = shared_zeros(v_dim)

        # We don't use the same learning rate for the different parts of the network
        # Hence we group them in different variables
        self.params_dico = {"W_p1": self.W_p1, "W_p2": self.W_p2, "W_p3": self.W_p3, "W_p4": self.W_p4,
                            "b_p1": self.b_p1, "b_p2": self.b_p2, "b_p3": self.b_p3, "b_p4": self.b_p4,
                            "W_o1": self.W_o1, "W_o2": self.W_o2, "W_o3": self.W_o3, "W_o4": self.W_o4,
                            "b_o1": self.b_o1, "b_o2": self.b_o2, "b_o3": self.b_o3, "b_o4": self.b_o4,
                            "W_z1": self.W_z1, "W_z2": self.W_z2, "W_z3": self.W_z3, "W_z4": self.W_z4,
                            "b_z1": self.b_z1, "b_z2": self.b_z2, "b_z3": self.b_z3, "b_z4": self.b_z4,
                            "W_enc1": self.W_enc1, "W_enc2": self.W_enc2, "W_enc3": self.W_enc3, "W_enc4": self.W_enc4,
                            "W_enc_mu": self.W_enc_mu, "W_enc_sig": self.W_enc_sig,
                            "b_enc1": self.b_enc1, "b_enc2": self.b_enc2, "b_enc3": self.b_enc3, "b_enc4": self.b_enc4,
                            "b_enc_mu": self.b_enc_mu, "b_enc_sig": self.b_enc_sig,
                            "W_prior1": self.W_prior1, "W_prior2": self.W_prior2, "W_prior3": self.W_prior3, "W_prior4": self.W_prior4,
                            "W_prior_mu": self.W_prior_mu, "W_prior_sig": self.W_prior_sig,
                            "b_prior1": self.b_prior1, "b_prior2": self.b_prior2, "b_prior3": self.b_prior3, "b_prior4": self.b_prior4,
                            "b_prior_mu": self.b_prior_mu, "b_prior_sig": self.b_prior_sig,
                            "W_dec1": self.W_dec1, "W_dec2": self.W_dec2, "W_dec3": self.W_dec3, "W_dec4": self.W_dec4,
                            "W_dec_bin": self.W_dec_bin,
                            "b_dec1": self.b_dec1, "b_dec2": self.b_dec2, "b_dec3": self.b_dec3, "b_dec4": self.b_dec4,
                            "b_dec_bin": self.b_dec_bin,
                            "L_oi": self.L_oi, "L_hi": self.L_hi, "b_i": self.b_i,
                            "L_oc": self.L_oc, "L_hc": self.L_hc, "b_c": self.b_c,
                            "L_of": self.L_of, "L_hf": self.L_hf, "b_f": self.b_f,
                            "L_oout": self.L_oout, "L_hout": self.L_hout, "b_out": self.b_out}

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
        self.debug_mode = True
        if self.debug_mode:
            # Since we mix shared variables and test_value,
            # we absolutely need to use the same dimensions for the test_values
            self.num_batches_test = 50  # We can choose any value here
            self.orch.tag.test_value = orch_debug
            self.piano.tag.test_value = piano_debug
        ####################################################################
        ####################################################################
        ####################################################################

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

    def inference(self, o, p, seq_length):
        # Infering z_t sequence from orch_t and piano_t
        #   (under both prior AND q distribution)

        # Initialize h_0 and c_0 states
        h_0 = T.zeros((self.h_dim,))
        c_0 = T.zeros((self.h_dim,))
        # Test values
        if self.debug_mode:
            h_0.tag.test_value = np.zeros((self.h_dim,), dtype=theano.config.floatX)
            c_0.tag.test_value = np.zeros((self.h_dim), dtype=theano.config.floatX)

        # Random mask for gaussian sampling
        epsilon = self.theano_rng.normal(size=(seq_length, self.z_dim), avg=0., std=1., dtype=theano.config.floatX)

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

        def inner_fn(o_t, p_t, epsilon_t, h_tm1, c_tm1):
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
            z_t = prior_mu_t + prior_sig_t * epsilon_t

            # Compute Z network
            z_1_t = propup_relu(z_t, self.W_z1, self.b_z1)
            z_2_t = propup_relu(z_1_t, self.W_z2, self.b_z2)
            z_3_t = propup_relu(z_2_t, self.W_z3, self.b_z3)
            z_4_t = propup_relu(z_3_t, self.W_z4, self.b_z4)

            # Compute new recurrent hidden state
            input_lstm = T.concatenate([o_t, z_4_t])
            h_t, c_t = self.lstm_prop(input_lstm, c_tm1, h_tm1)
            return h_t, c_t, enc_mu_t, enc_sig_t, prior_mu_t, prior_sig_t, z_4_t

        # Scan through input sequence
        ((h_seq, c_seq, enc_mu, enc_sig, prior_mu, prior_sig, z_4), updates) =\
            theano.scan(fn=inner_fn,
                        sequences=[o_4, p_4, epsilon],
                        outputs_info=[h_0, c_0, None, None, None, None, None])

        # Reconstruction from inferred z_t
        # Can be performed after scanning, which is computationally more efficient
        input_dec = T.concatenate([z_4, p_4, h_seq], axis=1)
        dec_1 = propup_relu(input_dec, self.W_dec1, self.b_dec1)
        dec_2 = propup_relu(dec_1, self.W_dec2, self.b_dec2)
        dec_3 = propup_relu(dec_2, self.W_dec3, self.b_dec3)
        dec_4 = propup_relu(dec_3, self.W_dec4, self.b_dec4)
        dec_bin = propup_sigmoid(dec_4, self.W_dec_bin, self.b_dec_bin)

        # We need :
        #   - prior : p(z_t) (w/ reparametrization trick, just pass mu and sigma)
        #   - approx inference : q(z_t|x_t)
        #   - reconstruction : p(x|z)
        return (enc_mu, enc_sig, prior_mu, prior_sig, dec_bin), updates

    def compute_nll_upper_bound(self, seq_length, validation=False):
        #############
        # Inference
        (enc_mu, enc_sig, prior_mu, prior_sig, dec_bin), updates = \
            self.inference(self.orch, self.piano, seq_length)

        #############
        # Cost
        dec_bin_non_zero = T.switch(dec_bin > 0, dec_bin, 1e-30)  # Avoid log zero
        recon = T.sum(T.nnet.binary_crossentropy(dec_bin_non_zero, self.orch), axis=1)
        # binary_crossentropy = nll for binary input. Sum along input dimension, mean along time (i.e. batch)
        # for real-valued units, use GaussianNLL
        kl = KLGaussianGaussian(enc_mu, enc_sig, prior_mu, prior_sig)
        # Mean over batches
        recon_term = T.mean(recon)
        kl_term = T.mean(kl)
        # Note that instead of maximazing the neg log-lik upper bound,
        # We here minimize the log-lik upper bound
        cost = recon_term + kl_term

        if not validation:
            #############
            # Gradient
            gparams = G.grad(cost, self.params_dico.values())
            #############
            # Updates
            updates_train = self.optimizer.get_updates(self.params_dico.values(), gparams, updates)
            #############
            # Cost
            return cost, updates_train
        else:
            return cost, recon_term, kl_term, dec_bin, updates

    def cost_updates(self, seq_length):
        import pdb; pdb.set_trace()
        cost, updates_train = self.compute_nll_upper_bound(seq_length=seq_length, validation=False)
        return cost, updates_train

    def validation(self, seq_length):
        # Validation = computing the nll upper bound on the validation set
        # So this function simply is a wrapper of cost_update function w/ no updates
        cost, recon_term, kl_term, dec_bin, updates_valid = self.compute_nll_upper_bound(seq_length=seq_length, validation=True)
        # Generate an orchestral sequence
        o, dec_bin, prior_mu, prior_sig, updates = self.generate(self.piano, seq_length)
        updates_valid.update(updates)
        # Compute the model accuracy
        # Note that this is really different from the accuracy for the cRBM, since it is evaluated on a sequence and
        # that the past orchestration is not re-initialized at each prediction
        accuracy = accuracy_measure(self.orch, o)
        return cost, recon_term, kl_term, dec_bin, accuracy, updates_valid

    def generation(self, seq_length):
        # seq_length must have the same length as self.piano.shape[0]
        o, _, _, _, updates = self.generate(self.piano, seq_length)
        return o, updates

    def generate(self, p, seq_length):
        # Genreate an orchestral sequence given a piano sequence

        # Initialize h_0 and c_0 states
        h_0 = T.zeros((self.h_dim,))
        c_0 = T.zeros((self.h_dim,))

        epsilon = self.theano_rng.normal(size=(seq_length, self.z_dim), avg=0., std=1., dtype=theano.config.floatX)

        # First, compute the piano parametrization
        # Piano network
        p_1 = propup_relu(p, self.W_p1, self.b_p1)
        p_2 = propup_relu(p_1, self.W_p2, self.b_p2)
        p_3 = propup_relu(p_2, self.W_p3, self.b_p3)
        p_4 = propup_relu(p_3, self.W_p4, self.b_p4)

        def inner_fn(p_t, epsilon_t, h_tm1, c_tm1):
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
            dec_bin_t = propup_sigmoid(dec_4, self.W_dec_bin, self.b_dec_bin)

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
                        sequences=[p_4, epsilon],
                        outputs_info=[h_0, c_0, None, None, None, None])

        return o, dec_bin, prior_mu, prior_sig, updates

    def dump_weights(self, directory='DEBUG/dump_weights/'):
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Write all the weights in a csv file
        for name, param in self.params_dico.items():
            filename = directory + name + ".csv"
            plot_param = param.get_value()
            # Check for NaN values
            if np.sum(np.isnan(plot_param)):
                print("NaN value for param " + name)
            # Round at 1e-5
            plot_param = np.round_(plot_param, decimals=5)
            np.savetxt(filename, plot_param, delimiter=",")
