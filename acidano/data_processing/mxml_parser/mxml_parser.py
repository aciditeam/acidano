#!/usr/bin/env python
# -*- coding: utf8 -*-
# A SAX-based parser for a MusicXML file written for multi-instrument scores
# with dynamics
# This is a minimalist yet exhaustive parser.
# It produces as an output a matrix corresponding to the piano-roll representation
#
# The file has to be parsed two time. A first time to get the total duration of the file.
#
# Idea : a minus amplitude at the end of a note (play the role of a flag saying : note stop here)
#
# TODO : Ajouter les <articulations> (notations DOSIM quoi wech)

import xml.sax
import os
import re

# Data import
import cPickle
from collections import OrderedDict

# Numpy
import numpy as np

# Acidano
from scoreToPianoroll import ScoreToPianorollHandler
from totalLengthHandler import TotalLengthHandler

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import mpldatacursor


def build_db(database_path, quantization, pitch_per_instrument, instru_dict_path=None, output_path='../Data/'):
    # First load the instrument dictionnary
    if instru_dict_path is None:
        # Create a defaut empty file if not indicated
        instru_dict_path = u"instru_regex.p"
        instru_dict = {}
    elif os.path.isfile(instru_dict_path):
        with open(instru_dict_path, "rb") as f:
            instru_dict = OrderedDict()
            instru_dict = cPickle.load(f)
    else:
        raise NameError(instru_dict_path + " is not a pickle file")

    # Get time length for the pianoroll
    T = 0
    for dirname, dirnames, filenames in os.walk(database_path):
        for filename in filenames:
            # Is it a music xml file ?
            filename_test = re.match("(.*)\.xml", filename, re.I)
            if not filename_test:
                continue

            full_path_file = os.path.join(dirname, filename)

            # Get the total length in quarter notes of the track
            pre_parser = xml.sax.make_parser()
            pre_parser.setFeature(xml.sax.handler.feature_namespaces, 0)
            Handler_length = TotalLengthHandler()
            pre_parser.setContentHandler(Handler_length)
            pre_parser.parse(full_path_file)
            total_length = Handler_length.total_length
            # Float number
            T += int(total_length)
    # Mutliply by the quantization
    T = T * quantization

    # Data are stored in a dictionnary
    N_instr = len(instru_dict.keys())
    orchestra_dim_full = N_instr * pitch_per_instrument
    pr_piano_full = np.zeros((T, pitch_per_instrument))
    pr_orchestra_full = np.zeros((T, orchestra_dim_full))
    arti_piano_full = np.zeros((T, pitch_per_instrument))
    arti_orchestra_full = np.zeros((T, orchestra_dim_full))
    new_track_ind = []

    # Build mapping
    instru_mapping = OrderedDict()
    counter = 0
    for key in instru_dict.keys():
        instru_mapping[key] = (counter, counter + pitch_per_instrument)
        counter = counter + pitch_per_instrument

    # Browse database_path folder
    time = 0
    for dirname, dirnames, filenames in os.walk(database_path):
        for filename in filenames:
            # Is it a music xml file ?
            filename_test = re.match("(.*)\.xml", filename, re.I)
            if not filename_test:
                continue

            full_path_file = os.path.join(dirname, filename)

            print "Parsing file : " + filename

            # Get the total length in quarter notes of the track
            pre_parser = xml.sax.make_parser()
            pre_parser.setFeature(xml.sax.handler.feature_namespaces, 0)
            Handler_length = TotalLengthHandler()
            pre_parser.setContentHandler(Handler_length)
            pre_parser.parse(full_path_file)
            total_length = Handler_length.total_length
            # Float number
            total_length = int(total_length)

            # Now parse the file and get the pianoroll, articulation and dynamics
            parser = xml.sax.make_parser()
            parser.setFeature(xml.sax.handler.feature_namespaces, 0)
            Handler_score = ScoreToPianorollHandler(quantization, instru_dict, total_length, pitch_per_instrument, False)
            parser.setContentHandler(Handler_score)
            parser.parse(full_path_file)

            # Using Mapping, build concatenated along time and pitch pianoroll
            track_length = total_length * quantization
            start_t = time
            end_t = time + track_length
            for instru_name, mat in Handler_score.pianoroll.iteritems():
                if instru_name == 'piano':
                    pr_piano_full[start_t:end_t, :] = mat
                else:
                    pitch_ind = instru_mapping[instru_name]
                    start_p = pitch_ind[0]
                    end_p = pitch_ind[1]
                    pr_orchestra_full[start_t:end_t, start_p:end_p] = mat
            for instru_name, mat in Handler_score.articulation.iteritems():
                if instru_name == 'piano':
                    arti_piano_full[start_t:end_t] = mat
                else:
                    pitch_ind = instru_mapping[instru_name]
                    start_p = pitch_ind[0]
                    end_p = pitch_ind[1]
                    arti_orchestra_full[start_t:end_t, start_p:end_p] = mat
            # Store starting track indices
            new_track_ind.append(time)

            # Increment time counter
            time += track_length

    # Get the list of non-zeros pitches : sum along time first
    reduction_mapping_orch = np.nonzero(np.sum(pr_orchestra_full, 0))
    reduction_mapping_piano = np.nonzero(np.sum(pr_piano_full, 0))
    # Strange behavior of numpy...
    reduction_mapping_orch = reduction_mapping_orch[0]
    reduction_mapping_piano = reduction_mapping_piano[0]
    # Remove unused pitches
    pr_orchestra = pr_orchestra_full[:, reduction_mapping_orch]
    arti_orchestra = arti_orchestra_full[:, reduction_mapping_orch]
    pr_piano = pr_piano_full[:, reduction_mapping_piano]
    arti_piano = arti_piano_full[:, reduction_mapping_piano]

    # Write the structure in a dictionary
    # See ~Recherche/Github/leo/Data/notes.md
    data = OrderedDict()
    data['quantization'] = quantization
    data['instru_mapping'] = instru_mapping
    data['reduction_mapping_orchestra'] = reduction_mapping_orch
    data['reduction_mapping_piano'] = reduction_mapping_piano
    data['articulation_orchestra'] = arti_orchestra
    data['articulation_piano'] = arti_piano
    data['pr_orchestra'] = pr_orchestra
    data['pr_piano'] = pr_piano
    data['change_track'] = new_track_ind

    with open(output_path, 'wb') as f:
        cPickle.dump(data, f)

if __name__ == '__main__':
    build_db(database_path='/Users/leo/Recherche/GitHub_Aciditeam/lop/Database/LOP_db_small',
             quantization=4,
             pitch_per_instrument=128,
             instru_dict_path='instru_regex.p',
             output_path='/Users/leo/Recherche/GitHub_Aciditeam/lop/Data/data.p')
